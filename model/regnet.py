import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from model.util import CalcDisp
from model.transformation import warp
from model.VTN import VTNAffineStem
from model.unet import UNet_Encoder, CrossAttentionModule, UNet_Decoder, CubicBSpline_Decoder, ShallowD3Net, UNet_New, CubicBSplineNet, SingleConv


class RegNet_DT_attention(nn.Module):
    def __init__(self, 
                nt,
                img_size = (96, 96)):
        super(RegNet_DT_attention, self).__init__()
        self.img_size = img_size
        self.nt = nt
        self.encoder1 = UNet_Encoder(ndim = 2,
                                     input_channels = self.nt, 
                                     enc_channels = (16, 32, 32))
        # encoder2 to create the displacement field
        self.encoder2 = UNet_Encoder(ndim = 2,
                                     input_channels = self.nt,
                                     enc_channels = (16, 32, 64, 64))
        self.cross_attention1 = CrossAttentionModule(input_channel1 = 32,
                                                    input_channel2 = 64,
                                                    positional_coding = 32)
        self.cross_attention2 = CrossAttentionModule(input_channel1 = 32,
                                                     input_channel2 = 64,
                                                     positional_coding = 32)
        self.CovMatHead = UNet_Decoder(ndim = 2,
                                    enc_channels = (16, 32, 32),
                                    dec_channels = (32, 16),
                                    conv_channels = (32, 16),# This is not applicable. 
                                    out_channels = 6,  
                                    conv_before_out = False)
        # self.B0Head = UNet_Decoder(ndim = 2,
        #                            enc_channels = (16, 32, 32),
        #                            dec_channels = (32,16),
        #                            conv_channels = (32, 16), # This is not applicable. 
        #                            out_channels = 1,
        #                            conv_before_out = False)
        self.BsplineEncoder = CubicBSpline_Decoder(ndim = 2,
                                                enc_channels = (16, 32, 64, 64),
                                                dec_channels = (64, 32, 32),
                                                resize_channels = (32, 16),
                                                out_channels = 2*self.nt,
                                                cps = (4,4),
                                                img_size = (96, 96),
                                                conv_before_out = False)
        self.spatial_transform = CubicBSplineFFDTransform(ndim = 2, 
                                            img_size = self.img_size,
                                            cps = (4,4),
                                            svf = True)
                                    
    def extract_rank1_along_nt(self, image_tensor):
        '''
        This function is to extract the rank-1 tensor along the nt dimension.
        input: image_tensor: nb, nt, nx, ny
        output: the rank1 image tensor: nb, 1, nx, ny
        '''
        nb, nt, nx, ny = image_tensor.shape
        rank1_images = torch.zeros_like(image_tensor)

        for b in range(nb):
            # Reshape the 3D tensor into a 2D matrix for each batch
            pixel_data = image_tensor[b].reshape(nt,-1)
            
            # Calculate the mean and center the data
            mean = torch.mean(pixel_data, dim=1, keepdim=True)
            pixel_data_centered = pixel_data - mean

            # Perform low-rank SVD
            U, S, V = torch.svd_lowrank(pixel_data_centered, q=1)
            S_matrix = torch.diag_embed(S)

            # Reconstruct the data
            reconstructed_data = torch.mm(U, torch.mm(S_matrix, V.T)) + mean

            # Reshape back to the original 3D shape
            rank1_images[b] = reconstructed_data.reshape(nt, nx, ny)
        #  take the mean of the dim 1
        # rank1_images = torch.mean(rank1_images, dim=1, keepdim=True)
        return rank1_images
    
    def forward(self, input_image_group):
        # input should be nbatch, nt(70), 96, 96 
        nb, nt, nx, ny = input_image_group.shape
        # D3Unet_out = self.D3Unet(input_image_group).reshape(nb, -1, nx, ny) 
        # use the low-rank of the output with nb*1*nx*ny, which mimics all the others 
        D3Unet_out = self.extract_rank1_along_nt(input_image_group) # nb, 1, nx, ny

        x1_enc_list = self.encoder1(D3Unet_out)
        x2_enc_list = self.encoder2(input_image_group.squeeze(1))
        x1_cross, x2_cross = self.cross_attention1(x1_enc_list[-2], 
                                                    x2_enc_list[-2])
        x1_enc_list[-2] = x1_cross
        x2_enc_list[-2] = x2_cross
        x1_cross, x2_cross = self.cross_attention2(x1_enc_list[-1],
                                                   x2_enc_list[-1])
        x1_enc_list[-1] = x1_cross
        x2_enc_list[-1] = x2_cross

        # if not initialized, there will be NaN in the output
        dt_tensor = self.CovMatHead(x1_enc_list)*0.002
        # b0 = self.B0Head(x1_enc_list)

        implicit_template = D3Unet_out

        flow_out = self.BsplineEncoder(x2_enc_list)
        flow, disp_i2t = self.spatial_transform(flow_out)
        warped_image = warp(input_image_group, disp_i2t)


        res = {'disp_i2t':disp_i2t, # nb, nt, 2, nx, ny
               'warped_image':warped_image,
               'template':implicit_template, # nb, 1, nx, ny
                'dt_tensor':dt_tensor}

        return res