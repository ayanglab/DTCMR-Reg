# conda activate py39
# here I specify the model path, wandb files is installed underneath
# also the datadir is the excel & MaskedOutputStrongDenoised folder

import importlib
import model.regnet, model.loss, model.util, utils.structure
from utils.visual import log_image_wandb_np, combine_flow_field_np, draw_image, draw_image_wiz_displacement
import torch, os
import numpy as np
import logging, tqdm
import wandb
from model.util import DTI_dataset_7dirs
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import argparse
import time

logging.basicConfig(level=logging.INFO, format = '%(levelname)s: %(message)s')

torch.autograd.set_detect_anomaly(True)

# Create the parser
parser = argparse.ArgumentParser(description="Model configuration")

# Add arguments
parser.add_argument("--dim", type=int, default=2, help="Dimension of the input image")
parser.add_argument("--times", type=int, default=70)
parser.add_argument("--epoch", type=int, default=5000)
parser.add_argument("--scale", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--smooth_reg", type=float, default=50)
parser.add_argument("--perce_reg", type=float, default=1e-4)
parser.add_argument("--dt_reg", type=int, default=200)
parser.add_argument("--dice_reg", type=int, default=2)
parser.add_argument("--load_optimizer", type=bool, default=False)
parser.add_argument("--debug", action='store_true', default=False, help="Enable debug mode")
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--vali_epoch", type=int, default=50)
parser.add_argument("--np_var", type=str, default="denoised_data") # or bimgs
parser.add_argument("--model_dir", type=str, default="/media/ssd/Models/")
parser.add_argument("--data_dir", type=str, default="/media/ssd/ISMRM/")


# Parse the arguments
args = parser.parse_args()

# Use the arguments
config = dict(
    dim=args.dim,
    times=args.times,
    epoch=args.epoch,
    scale=args.scale,
    learning_rate=args.lr,
    smooth_reg=args.smooth_reg,
    perce_reg=args.perce_reg,
    dt_reg=args.dt_reg,
    dice_reg=args.dice_reg,
    load_optimizer=args.load_optimizer,
    debug=args.debug,
    batch_size=args.batch_size,
    log=args.log,
    vali_epoch=args.vali_epoch,
    np_var=args.np_var,
    model_dir=args.model_dir,
    data_dir=args.data_dir
)

config = utils.structure.Struct(**config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if config.log:
    wandb.login(key="000")
    wandb_name = 'DICE_' + str(config.dice_reg) + '_' + config.np_var + '_B0asS0LR_' + time.strftime("%Y%m%d")
    logger = wandb.init(project="000",
                        name = wandb_name, 
                        config=config, 
                        resume="allow", 
                        entity="000", 
                        dir=config.model_dir)

states_folder = config.model_dir
saved_folder = 'DICE_' + str(config.dice_reg) + '_' + config.np_var +'_B0asS0_LRInputEncoder_smooth' + str(config.smooth_reg) + '__perce' + str(config.perce_reg) + '__dt' + str(config.dt_reg) + '__dice' + str(config.dice_reg) 
print(saved_folder)
new_states_folder = os.path.join(states_folder, saved_folder)

if not os.path.exists(os.path.join(states_folder, saved_folder)):
    os.makedirs(new_states_folder)

rootpath = os.path.join(config.data_dir, 'MaskedOutputStrongDenoised')
dataset = DTI_dataset_7dirs(rootdir = rootpath, 
                            exceldir = os.path.join(config.data_dir, 'DTCMR_cases.xlsx'),
                            np_var = config.np_var)

if config.debug:
    num_data = 5
    dataset = torch.utils.data.Subset(dataset, np.arange(num_data))
    train_loader = DataLoader(dataset, config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset, config.batch_size, shuffle=False, num_workers=2)
    config.batch_size = 1
else:
    num_data = len(dataset)
    val_percent = 0.2
    num_val = int(num_data*val_percent)
    num_train = num_data - num_val

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    # You can then use these datasets with DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

image_shape = [96, 96] # (d, h, w)

regnet = model.regnet.RegNet_DT_attention(nt = config.times, img_size = image_shape)
regnet = regnet.to(device)

DICELoss = model.loss.DiceSC()

DTLoss = model.loss.MILossGaussianGroup(vmin =  0.0,
                                        vmax =  1.0,
                                        num_bins =  64,
                                        sample_ratio = 0.1,
                                        normalised = True)
DTModelLoss = model.loss.DTGroupModelBased(DTLoss)

# all the layers in the perceptual loss
target_layers = np.arange(20,30).tolist()
target_layers = [str(i) for i in target_layers]
perceptual_loss = model.loss.PerceptualLoss(target_layers)

optimizer = torch.optim.Adam(regnet.parameters(), lr = config.learning_rate)

grid_tuple = [np.arange(grid_length, dtype = np.float32) for grid_length in image_shape]

diff_stats = []
pbar = tqdm.tqdm(range(config.epoch))
for epoch in pbar:
    total_epoch_loss = 0.
    
    for batches, input_dict in enumerate(train_loader):
        input_image = input_dict['normed_data'].to(device)
        b_values = input_dict['bvalue'].to(device)
        dirs = input_dict['dirs4PC'].to(device)
        S0 = input_dict['rank1img4b0'].to(device).unsqueeze(1)
        batch, nt, ny, nx = input_image.shape
        # I want to just put the mask in the loss part, for testing, no need to input the mask.
        res = regnet(input_image)
        
        total_loss = 0
        # ATT! the covariance matrix is about 0.003 max and -0.001 min
        dt_loss, y_warped, y_generated, b0_img, dt_tensor = DTModelLoss.loss(res['warped_image'],
                                                                             S0,
                                                                             res['dt_tensor'],
                                                                             b_values,
                                                                             dirs)
        # recovered to the original scale
        total_loss += dt_loss * config.dt_reg
        dt_loss = dt_loss
        

        smooth_loss = model.loss.l2reg_loss_group(res['disp_i2t'])
        total_loss += config.smooth_reg*smooth_loss
        smooth_loss_item = smooth_loss.item()
        
        perce_loss = perceptual_loss(res['warped_image'], y_generated.float())
        total_loss += config.perce_reg*perce_loss
        perceptual_loss_item = perce_loss.item()

        dice_loss, warped_seg = DICELoss.loss(input_dict['mask'],input_dict['template_mask'],res['disp_i2t'])
        total_loss += dice_loss * config.dice_reg
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    if epoch % config.vali_epoch == 0:   
        with torch.no_grad():  
            for batch_val, input_dict_val in enumerate(val_loader):
                images_val = input_dict_val['normed_data'].to(device)
                bvalues_val = input_dict_val['bvalue'].to(device)
                dirs_val = input_dict_val['dirs4PC'].to(device)
                S0_val = input_dict_val['rank1img4b0'].to(device).unsqueeze(1)
                batch, nt, ny, nx = images_val.shape                
                
                res_val = regnet(images_val)
                
                total_loss_val = 0.
                dt_loss_val, y_warped_val, y_generated_val, b0_img_val, dt_tensor_val = DTModelLoss.loss(res_val['warped_image'], 
                                                                                                         S0_val,
                                                                                                         res_val['dt_tensor'],
                                                                                                         bvalues_val,
                                                                                                         dirs_val)
                # recovered to the original scale
                total_loss_val += dt_loss_val * config.dt_reg

                smooth_loss_val = model.loss.l2reg_loss_group(res_val['disp_i2t'])
                total_loss_val += config.smooth_reg*smooth_loss_val
                smooth_loss_item_val = smooth_loss_val.item()
                
                perce_loss_val = perceptual_loss(res_val['warped_image'], y_generated_val.float())
                total_loss_val += config.perce_reg*perce_loss_val
                perceptual_loss_val_item = perce_loss_val.item()

                # for simplicity, I use MI for the groupwise-segmentation loss
                dice_loss_val, warped_seg_val = DICELoss.loss(input_dict_val['mask'],input_dict_val['template_mask'], res_val['disp_i2t'])
                total_loss_val += dice_loss_val * config.dice_reg
                
            if config.log:
                # log both training and validation
                # first draw the training images
                randomslice = 0
                # random choose 10 frames
                randframe = np.random.randint(0, nt, 10)
                imgb4 = input_image[randomslice,randframe].squeeze(0).detach().cpu().numpy()
                
                segb4 = input_dict['mask'][randomslice,randframe].squeeze(0).detach().cpu().numpy()
                segafter = warped_seg[randomslice,randframe].squeeze(0).detach().cpu().numpy()

                show_warped = y_warped[randomslice,randframe].squeeze(0).detach().cpu().numpy()
                show_synt = y_generated[randomslice,randframe].squeeze(0).detach().cpu().numpy()
                
                b0_slice = S0[randomslice].detach().cpu().numpy()
                dtslice = dt_tensor[randomslice].detach().cpu().numpy()
                
                show_disp = combine_flow_field_np(res['disp_i2t'][randomslice,randframe].squeeze(0).detach().cpu().numpy())
                # first log the epoch of training
                logger.log({'trn/DTloss': dt_loss*config.dt_reg,
                    'trn/smooth_loss': smooth_loss_item*config.smooth_reg,
                    'trn/dice_loss': dice_loss*config.dice_reg, 
                    'trn/total_loss': total_loss.item(),
                    'trn/perceptual_loss': perceptual_loss_item*config.perce_reg,
                    'epoch': epoch})
                
                logger.log({'val/DTloss': dt_loss_val*config.dt_reg,
                    'val/smooth_loss': smooth_loss_item_val*config.smooth_reg,
                    'val/dice_loss': dice_loss_val* config.dice_reg,
                    'val/total_loss': total_loss_val.item(),
                    'val/perceptual_loss': perceptual_loss_val_item*config.perce_reg,
                    'epoch': epoch})

                draw_image_wiz_displacement(images = [imgb4, show_warped, show_synt], 
                                            disp = res['disp_i2t'][randomslice,randframe].detach().cpu().numpy(), 
                                            name = 'trn/images',
                                            vmin = -0.5, 
                                            vmax = 1, 
                                            scale = config.scale)
                draw_image_wiz_displacement(images = [segb4, segafter], 
                                            disp = res['disp_i2t'][randomslice,randframe].detach().cpu().numpy(), 
                                            name = 'trn/seg', 
                                            vmin = 0, 
                                            vmax = 1, 
                                            scale = config.scale)
                log_image_wandb_np(images = [show_disp], 
                                name = 'trn/disp')
                draw_image(images = [b0_slice], 
                        name = 'trn/S0', 
                        vmin = 0, 
                        vmax = 0.7)

                draw_image(images = [dtslice], 
                        name = 'trn/M0', 
                        vmin = -0.001, 
                        vmax = 0.002)
            
                imgb4_val = images_val[randomslice, randframe].squeeze(0).detach().cpu().numpy()
                
                segb4_val = input_dict_val['mask'][randomslice, randframe].squeeze(0).detach().cpu().numpy()
                segafter_val = warped_seg_val[randomslice, randframe].squeeze(0).detach().cpu().numpy()

                show_warped_val = y_warped_val[randomslice, randframe].squeeze(0).detach().cpu().numpy()
                show_synt_val = y_generated_val[randomslice, randframe].squeeze(0).detach().cpu().numpy()

                b0_slice_val = S0_val[randomslice].detach().cpu().numpy()
                dtslice_val = dt_tensor_val[randomslice].detach().cpu().numpy()
                
                show_disp_val = combine_flow_field_np(res_val['disp_i2t'][randomslice, randframe].squeeze(0).detach().cpu().numpy())


                draw_image_wiz_displacement(images = [imgb4_val, show_warped_val, show_synt_val], 
                                            disp = res_val['disp_i2t'][randomslice,randframe].detach().cpu().numpy(), 
                                            name = 'val/images',
                                            vmin = -0.5, 
                                            vmax = 1, 
                                            scale = config.scale)
                draw_image_wiz_displacement(images = [segb4_val, segafter_val], 
                                            disp = res_val['disp_i2t'][randomslice,randframe].detach().cpu().numpy(), 
                                            name = 'val/seg', 
                                            vmin = 0, 
                                            vmax = 1, 
                                            scale = config.scale)
                log_image_wandb_np(images = [show_disp_val], 
                                    name = 'val/disp')
                draw_image(images = [b0_slice_val], 
                            name = 'val/S0', 
                            vmin = 0, 
                            vmax = 0.7)
                draw_image(images = [dtslice_val], 
                            name = 'val/M0', 
                            vmin = -0.001, 
                            vmax = 0.002)
        # save the model
        states = {'config': config, 'model': regnet.state_dict(), 'optimizer': optimizer.state_dict()}
        states_file = f'DT_epoch_{epoch}.pth'
        torch.save(states, os.path.join(new_states_folder, states_file))
        logging.info(f'save model and optimizer state {states_file}')
