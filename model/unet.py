import torch
from torch import nn
import torch.nn.functional as F
from model.misc import param_ndim_setup
import math
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class UNet_Encoder(nn.Module):
    """
    Adpated from the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 ndim, 
                 input_channels = 2, 
                 enc_channels=(16, 32, 32, 32),
                 ):
        '''
        args:
            ndim: dimension of convolution
            input_channels: number of input channels
            enc_channels: number of channels in the encoder
            conv_channels: number of channels in the conv layers before output
        '''
        super(UNet_Encoder, self).__init__()

        self.ndim = ndim
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = input_channels if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    convNd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                    nn.LeakyReLU(0.2))
            )

    def forward(self, x):
        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))             
        return fm_enc


class UNet_Decoder(nn.Module):
    """
    Adpated from the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 ndim, 
                 enc_channels = (16, 32, 64),
                 dec_channels = (32, 32, 32, 32),
                 conv_channels = (16, 16),
                 out_channels = 4,
                 conv_before_out=True
                 ):
        '''
        args:
            ndim: dimension of convolution
            input_channels: number of input channels
            enc_channels: number of channels in the encoder
            conv_channels: number of channels in the conv layers before output
        '''
        super(UNet_Decoder, self).__init__()

        self.ndim = ndim
        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i-1] + enc_channels[-i-1]
            self.dec.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        if conv_before_out:
            self.out_layers = nn.ModuleList()
            for i in range(len(conv_channels)):
                in_ch = dec_channels[-1] + enc_channels[0] if i == 0 else conv_channels[i-1]
                self.out_layers.append(
                    nn.Sequential(
                        convNd(ndim, in_ch, conv_channels[i], a=0.2),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )
            # final prediction layer with additional conv layers
            self.out_layers.append(
                convNd(ndim, conv_channels[-1], out_channels)
            )
        else:
            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                convNd(ndim, dec_channels[-1] + enc_channels[0], out_channels)
            )
            
    def forward(self, enc_out):
        # encoder
        dec_out = enc_out[-1]
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, enc_out[-2-i]], dim=1)
        y = dec_out
        for out_layer in self.out_layers:
            y = out_layer(y)                   
        return y

class CubicBSpline_Decoder(UNet_Decoder):
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 resize_channels=(32, 16),
                 out_channels = 2,
                 cps=(4, 4),
                 img_size=(160, 160),
                 conv_before_out=True
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(CubicBSpline_Decoder, self).__init__(ndim = ndim,
                                              enc_channels = enc_channels,
                                              dec_channels = dec_channels,
                                              out_channels = out_channels,
                                              conv_before_out=conv_before_out)

        # determine and set output control point sizes from image size and control point spacing
        img_size = param_ndim_setup(img_size, ndim)
        cps = param_ndim_setup(cps, ndim)
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
                                  for imsz, c in zip(img_size, cps)])
        # Network:
        # encoder: same u-net encoder
        # decoder: number of decoder layers / times of upsampling by 2 is decided by cps
        num_dec_layers = 4 - int(math.ceil(math.log2(min(cps))))
        self.dec = self.dec[:num_dec_layers]

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                if num_dec_layers > 0:
                    in_ch = dec_channels[num_dec_layers-1] + enc_channels[-num_dec_layers-1]
                else:
                    in_ch = enc_channels[-1]
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndim, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # final prediction layer
        delattr(self, 'out_layers')  # remove u-net output layers
        self.out_layer = convNd(ndim, 
                                resize_channels[-1], 
                                out_channels)

    def forward(self, enc_out):
        # decoder: conv + upsample + concatenate skip-connections
        if len(self.dec) > 0:
            dec_out = enc_out[-1]
            for i, dec in enumerate(self.dec):
                dec_out = dec(dec_out)
                dec_out = self.upsample(dec_out)
                dec_out = torch.cat([dec_out, enc_out[-2-i]], dim=1)
        else:
            dec_out = enc_out

        # resize output of encoder-decoder
        x = interpolate_(dec_out, size=self.output_size)
        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        y = self.out_layer(x)
        
        y = y.view(y.shape[0], y.shape[1]//2, 2, y.shape[2], y.shape[3])
        return y

def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size = 3,
           stride = 1,
           padding = 1,
           a=0.):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation

    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f"Conv{ndim}d")(in_channels = in_channels,
                                          out_channels = out_channels,
                                          kernel_size = kernel_size,
                                          stride = stride,
                                          padding = padding)
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd
 
class AttentionDownsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 positional_coding = 32):
        super(AttentionDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              positional_coding,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1)
    def forward(self, x):
        x_pool = F.max_pool2d(input = x,
                        kernel_size = (3,3),
                        stride = (2,2),
                        padding = (1,1))
        x_conv = self.conv(x_pool)
        return x_conv

class AttentionUpsample(nn.Module):
    def __init__(self,
                 positional_coding: 32,
                 in_channels: int):
        super(AttentionUpsample, self).__init__()
        self.conv = nn.Conv2d(positional_coding,
                              in_channels,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        x_up = self.upsample(x)
        x_conv = self.conv(x_up)
        return x_conv
             
class CrossAttentionModule(nn.Module):
    def __init__(self, 
                 input_channel1: int, # channel of the first layer
                 input_channel2: int, # channel of the second layers
                 positional_coding = 32):
        super(CrossAttentionModule, self).__init__()
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2
        self.positional_coding = positional_coding
        # 2 layers -> downsample -> conv with the same channel  -> concate and conv
        self.downsample1 = AttentionDownsample(input_channel1, positional_coding)
        self.downsample2 = AttentionDownsample(input_channel2, positional_coding)
        # position encoding -> upsample -> same dim as the input
        self.upsample1 = AttentionUpsample(positional_coding, input_channel1)
        self.upsample2 = AttentionUpsample(positional_coding, input_channel2)
        
        self.conv1 = nn.Conv2d(input_channel1*2,
                                 input_channel1,
                                 kernel_size = 3,
                                 stride = 1,
                                 padding = 1)
        self.conv2 = nn.Conv2d(input_channel2*2,
                                input_channel2,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1)
    def forward(self, x1, x2):
        '''
        x1: (batch_size, c1, h1, w1)
        x2: (batch_size, c2, h2, w2)
        '''
        nb, nc1, nh1, nw1 = x1.shape
        nb, nc2, nh2, nw2 = x2.shape
        
        # Flatten the feature maps
        x1_feature = self.downsample1(x1).view(nb, self.positional_coding, -1) # nb, position, 6400
        x2_feature = self.downsample2(x2).view(nb, self.positional_coding, -1) # nb, position, 1600
        
        # Calculate dot product for cross-weights
        cross_weights = torch.bmm(x1_feature.transpose(1,2), x2_feature)
        # Normalize cross-weights and reshape to match input dimensions
        cross_weights = F.softmax(cross_weights, dim=-1) # 2*6400*1600
        
        x1cross = torch.bmm(x2_feature, cross_weights.transpose(1,2)).view(nb, self.positional_coding, nh1//2, nw1//2)
        x2cross = torch.bmm(x1_feature, cross_weights).view(nb, self.positional_coding,nh2//2, nw2//2)
        
        x2_up = self.upsample2(x2cross)
        x1_up = self.upsample1(x1cross)
        # concate with the original input, and conv to get the output
        x1cross = torch.cat((x1_up, x1), dim=1)
        x2cross = torch.cat((x2_up, x2), dim=1)
        
        x1out = self.conv1(x1cross)
        x2out = self.conv2(x2cross)
        return x1out, x2out


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        elif ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      )
    return y

