import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.ndimage
import numpy as np
import math
import torchvision.models as models
import torchvision.transforms as transforms
from model.transformation import warp

torch.backends.cudnn.deterministic = True


class PerceptualLoss(nn.Module):
    def __init__(self, target_layers):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()  # Set to evaluation mode
        for param in vgg.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            vgg = vgg.cuda()
        self.vgg = vgg
        self.target_layers = target_layers
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        input_3_channels = self.adapt_channels(input)
        target_3_channels = self.adapt_channels(target)
        
        input_features = self.get_features(input_3_channels)
        target_features = self.get_features(target_3_channels)
        loss = 0
        for input_feature, target_feature in zip(input_features, target_features):
            loss += self.loss(input_feature, target_feature)
        return loss

    def get_features(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.target_layers:
                features.append(x)
        return features
    
    def adapt_channels(self, x):
        # Example: Mean across channels to condense to 3 channels
        # This is a naive approach, adjust according to your needs
        x_mean = x.mean(dim=1, keepdim=True)
        return torch.cat([x_mean, x_mean, x_mean], dim=1)


class DTGroupModelBased():
    # This works for the mutual info of the DT maps and the warped images.
    def __init__(self, MILoss):
        super(DTGroupModelBased, self).__init__()
        self.mi_loss = MILoss

    def get_B_matrix_wizbatch(self,
                              b: torch.Tensor, # (nb, nt)
                             G: torch.Tensor): # (nb, nt, 3)
        _B = torch.stack([
                -b * G[..., 0] ** 2,
                -b * G[..., 1] ** 2,
                -b * G[..., 2] ** 2,
                -2 * b * G[..., 0] * G[..., 1],
                -2 * b * G[..., 0] * G[..., 2],
                -2 * b * G[..., 1] * G[..., 2]
            ], dim=-1)
        return _B
    
    def norm2float_batch(self,img):
        min_vals = torch.min(img.view(img.shape[0], -1), dim=1, keepdim=True)[0][:, :, None, None]
        max_vals = torch.max(img.view(img.shape[0], -1), dim=1, keepdim=True)[0][:, :, None, None]

        # Avoid division by zero
        epsilon = 1e-8
        normalized_img = (img - min_vals) / (max_vals - min_vals + epsilon)
        return normalized_img

    def loss(self, 
             y_warped,
             b0_img,
             dt_tensor,
             b_value,
             dirs):
        """
        y_warped: [nb, nt, nx, ny]
        parameters: b0_img: [nb, 1, nx, ny] + dt_tensor: [nb, 6, nx, ny]
        b_value: [nb, nt]
        dirs: [nb, nt, 3]
        """
        device = b0_img.device #[B,7,160,160]
        
        nb, nn, nx, ny = b0_img.shape
        nt = b_value.shape[1]
        D6 = dt_tensor.reshape((nb, 6, nx*ny)).unsqueeze(-1).repeat(1, 1, 1, nt). permute(0,3,2,1) 
        
        _B = self.get_B_matrix_wizbatch(b_value, dirs)
        B = _B.unsqueeze(-1).repeat(1, 1, 1, nx*ny).permute(0, 1, 3, 2) # [nb, nt, nxny, 6]
        B = B.float()
        D6 = D6.float()

        temp= torch.einsum("abcde,abcef->abc", B.unsqueeze(-1).permute(0, 1, 2, 4, 3), D6.unsqueeze(-1)) # [nb, nt, HW]

        SI = b0_img.repeat(1,nt,1,1) * torch.exp(temp.reshape((nb, nt, nx, ny))) # [nb, nt, HW, 1]
        
        y_generated = self.norm2float_batch(SI)
        # y_generated = SI 
   
        loss = self.mi_loss(y_warped.float(), y_generated.float())

        return loss, y_warped, y_generated, b0_img, dt_tensor

class DiceSC():
    def __init__(self):
        v=1

    def loss(self,
             y_prev,
             y_true,
             disp_t, 
             printflag=False):
        '''
        This function calculate the DICE between: 
        y_pred: nb, nt, nx, ny  with the input mask
        y_true: nb, 1, nx, ny with the template
        disp_t: nb, nt, 2, nx, ny with the displacement field

        '''
        # Check if y_true needs to be expanded to match y_pred's dimensions
        if y_true.size(1) == 1:
            y_true = y_true.expand_as(y_prev)

        device = y_prev.device

        y_true = y_true.to(device)

        warped_seg = warp(y_prev, disp_t)

        # Initialize an empty tensor to store the dice loss for each frame
        dice_losses = torch.zeros(y_prev.size(1))  # nt is the size of the second dimension

        # Calculate dice loss for each frame
        for frame_idx in range(warped_seg.size(1)):  # Iterate over the 'nt' dimension
            y_pred_flat = warped_seg[:, frame_idx].contiguous().view(-1)
            y_true_flat = y_true[:, frame_idx].contiguous().view(-1)

            intersection = (y_pred_flat * y_true_flat).sum()
            union = y_pred_flat.sum() + y_true_flat.sum()

            dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1 - dice_score
            dice_losses[frame_idx] = dice_loss

            if printflag:
                print(f"Frame {frame_idx} Dice Loss: {dice_loss.item()}")
        # Return the mean dice loss across all frames
        return dice_losses.sum(), warped_seg


class MILossGaussianGroup(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(MILossGaussianGroup, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))

        Returns:
            (Normalise)MI: (scalar)
        """
        # FW: reshape the input tensors to (nb*nt, 1, nx, ny)
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        y = y.reshape(-1, 1, y.shape[-2], y.shape[-1])
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            # x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            # y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]
            x = x.reshape(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.reshape(y.size()[0], 1, -1)[:, :, idx_choice]
            

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)

def l2reg_loss_group(u):
    """L2 regularisation loss"""
    derives = []
    # here u is the displacement field with size of [nbatch, 2, nx, ny]
    u = u.reshape(-1, 2, u.shape[-2], u.shape[-1])
    ndim = u.size()[1]
    for i in range(ndim):
        # calculate the first order derivative (along x/y dir)
        derives += [finite_diff(u, dim=i)]
    # L2 loss on the derivative. 
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss



def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Input shape (N, ndim, *sizes), mode='foward', 'backward' or 'central'"""
    assert type(x) is torch.Tensor
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    if mode == "central":
        # TODO: implement central difference by 1d conv or dialated slicing
        raise NotImplementedError("Finite difference central difference mode")
    else:  # "forward" or "backward"
        # configure padding of this dimension
        paddings = [[0, 0] for _ in range(ndim)]
        if mode == "forward":
            # forward difference: pad after
            paddings[dim][1] = 1
        elif mode == "backward":
            # backward difference: pad before
            paddings[dim][0] = 1
        else:
            raise ValueError(f'Mode {mode} not recognised')

        # reverse and join sublists into a flat list (Pytorch uses last -> first dim order)
        paddings.reverse()
        paddings = [p for ppair in paddings for p in ppair]

        # pad data
        if boundary == "Neumann":
            # Neumann boundary condition
            x_pad = F.pad(x, paddings, mode='replicate')
        elif boundary == "Dirichlet":
            # Dirichlet boundary condition
            x_pad = F.pad(x, paddings, mode='constant')
        else:
            raise ValueError("Boundary condition not recognised.")

        # slice and subtract
        x_diff = x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) \
                 - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

        return x_diff
