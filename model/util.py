import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import pandas as pd
import scipy.io as sio

class DTI_dataset_7dirs(data.Dataset):
    def __init__(self,
                 rootdir: str,
                 device:  str = 'cuda',
                 mode: str = 'train',
                 exceldir: str = '/media/data/DTCMR_cases.xlsx',
                 np_var: str = 'denoised_data'):
        super(DTI_dataset_7dirs, self).__init__()
        self.rootdir = rootdir
        self.device = device
        # subfile ends with mat
        self.submat = [f for f in sorted(os.listdir(self.rootdir)) if f.endswith('.mat')]
        # if the df['denoise'] is larger than 4, then append the case to the list
        df = pd.read_excel(exceldir)
        self.denoised_cases = []
        for i in range(len(df)):
            if df['denoised'][i] > 3:
                self.denoised_cases.append(df['Case'][i])
                # since random, append twice.
                self.denoised_cases.append(df['Case'][i])
        self.slices = 70
        self.mode = mode
        self.np_var = np_var

    def __len__(self):
        return len(self.denoised_cases)
    
    def __getitem__(self, index):
        DTInfo = {}
        # load the mat files and get the denoised data, bvalues and directions
        # 'bimgs', 'bvalue', 'denoised_data', 'dirs', 'dirs4PC', 'mask'
        mat = sio.loadmat(os.path.join(self.rootdir, self.denoised_cases[index]))
        # save the max and min 
        # get the central 96*96
        # random select the 70 slices
        origin_slices = mat[self.np_var].shape[2]
        dirs = mat['dirs4PC']

        # Process the directions and categorize into unique groups
        rounded_dirs = np.round(dirs, 1)
        unique_rows, inverse_indices = np.unique(rounded_dirs, axis=0, return_inverse=True)

        # Initialize dictionary to store indices for each unique rounded row
        dynamic_category_indices = {tuple(row): [] for row in unique_rows}
        for i, idx in enumerate(inverse_indices):
            dynamic_category_indices[tuple(rounded_dirs[i])].append(i)

        # Shuffle the categories for even distribution
        categories = list(dynamic_category_indices.keys())
        np.random.shuffle(categories)

        # Initialize list to store selected slice indices
        slice_idx = []

        # Loop until you have 70 slices, cycling through categories
        while len(slice_idx) < 70:
            for category in categories:
                if len(slice_idx) >= 70:
                    break  # Stop if we've reached the desired number of slices
                if dynamic_category_indices[category]:  # Check if the category still has indices left
                    # Randomly select an index from this category
                    selected_index = np.random.choice(dynamic_category_indices[category], replace=False)
                    # Add the selected index to the list
                    slice_idx.append(selected_index)
                    # Remove the selected index from the category to avoid re-selection
                    dynamic_category_indices[category].remove(selected_index)

        rotate = mat['rotate'][0,0]
        denoised_data = mat[self.np_var][80:(96+80),...,slice_idx]
        # for the references data, normalize them 
        denoised_data = denoised_data.astype(np.float32)
        nmax = denoised_data.max()
        nmin = denoised_data.min()
        normed_data = (denoised_data - nmin)/(nmax - nmin)

        # select the 70 slices (70 is the smallest)
        DTInfo['normed_data'] = np.transpose(normed_data, [2,0,1]) # get the nslice*96*96
        DTInfo['bvalue'] = mat['bvalue'][0,slice_idx] # :nslice
        DTInfo['dirs4PC'] = mat['dirs4PC'][slice_idx,:] # (:nslice, 3)

        # get the index of b-value that is lower than 100
        b0idx = np.where(DTInfo['bvalue'] < 100)
        # if no b0, then select the bvalue < 200
        if b0idx[0].shape[0] == 0:
            b0idx = np.where(DTInfo['bvalue'] < 200)
        
        rank1img = mat['rank1img'][...,b0idx[0]]
        # took the average of the rank1img
        # normalize the rank1img
        rank1img_norm = (rank1img - nmin)/(nmax - nmin)
        DTInfo['rank1img4b0'] = np.mean(rank1img_norm, axis=-1)[80:(96+80)]

        # for all masks, make the gray with 1 and others 0.
        DTInfo['mask'] = np.transpose(mat['mask'][80:(96+80),...,slice_idx], [2,0,1]) # (nslice*96*96)
        DTInfo['mask'][DTInfo['mask']>1] = 0
        DTInfo['max'] = nmax
        DTInfo['min'] = nmin
        # for all masks, make the gray with 1 and others 0.
        DTInfo['template_mask'] = mat['template_mask'][:,80:(96+80)]
        DTInfo['template_mask'][DTInfo['template_mask']>1] = 0
        if rotate == 1:
            # transpose the data if the orientation is not correct to match different bvalues.
            DTInfo['normed_data'] = np.transpose(DTInfo['normed_data'], [0,2,1])
            DTInfo['mask'] = np.transpose(DTInfo['mask'], [0,2,1])
            DTInfo['template_mask'] = np.transpose(DTInfo['template_mask'], [0,2,1])
            DTInfo['rank1img4b0'] = np.transpose(DTInfo['rank1img4b0'], [1,0])
        if self.mode == 'test':
            DTInfo['subject'] = self.denoised_cases[index]
        return DTInfo
    
# load the DTI dataset, load the 1*7*10 data with dir and b-value
# if it is test mode, save the subject name in the dict. 
class DTI_dataset_test(data.Dataset):
    def __init__(self,
                 rootdir: str,
                 device:  str = 'cuda',
                 np_var: str = 'denoised_data',
                 mode: str = 'test'):
        super(DTI_dataset_test, self).__init__()
        self.rootdir = rootdir
        self.device = device
        # subfile ends with mat
        self.submat = [f for f in sorted(os.listdir(self.rootdir)) if f.endswith('.mat')]
        # if the df['denoise'] is larger than 4, then append the case to the list
        exceldir = '/media/data/DTCMR_cases.xlsx'
        self.np_var = np_var
        df = pd.read_excel(exceldir)
        self.denoised_cases = []
        #TODO: some failed cases may exist.
        for i in range(len(df)):
            # for test, if the rank is smaller than 4, then use it as test. 
            if df['denoised'][i] < 4 and df['denoised'][i] > 0:
                self.denoised_cases.append(df['Case'][i])
        self.slices = 70
        self.mode = mode

    def __len__(self):
        return len(self.denoised_cases)
    
    def __getitem__(self, index):
        DTInfo = {}
        # load the mat files and get the denoised data, bvalues and directions
        # 'bimgs', 'bvalue', 'denoised_data', 'dirs', 'dirs4PC', 'mask'
        mat = sio.loadmat(os.path.join(self.rootdir, self.denoised_cases[index]))
        # save the max and min 
        # get the central 96*96
        rotate = mat['rotate'][0,0]
        origin_slices = mat[self.np_var].shape[2]
        # randomly select the slices
        # make sure that the acquired the data is less then 140 frames, output the data as 2 batch with the idx.
        denoised_data_1 = mat[self.np_var][80:(96+80),...,:self.slices]
        denoised_data_2 = mat[self.np_var][80:(96+80),...,-self.slices:]
        # get the index of -self.slices:
        idx2d = origin_slices - self.slices
        # for the references data, normalize them 
        denoised_data_1 = denoised_data_1.astype(np.float32)
        nmax_1, nmin_1 = denoised_data_1.max(),denoised_data_1.min()
        normed_data_1 = (denoised_data_1 - nmin_1)/(nmax_1 - nmin_1)

        denoised_data_2 = denoised_data_2.astype(np.float32)
        nmax_2, nmin_2 = denoised_data_2.max(),denoised_data_2.min()
        normed_data_2 = (denoised_data_2 - nmin_2)/(nmax_2 - nmin_2)

        # select the 70 slices (70 is the smallest)
        DTInfo['normed_data1'] = np.transpose(normed_data_1, [2,0,1]) # get the nslice*96*96
        DTInfo['bvalue1'] = mat['bvalue'][0,:self.slices] # :nslice
        DTInfo['dirs4PC1'] = mat['dirs4PC'][:self.slices,:] # (:nslice, 3)
        # for all masks, make the gray with 1 and others 0.
        DTInfo['mask1'] = np.transpose(mat['mask'][80:(96+80),...,:self.slices], [2,0,1]) # (nslice*96*96)
        DTInfo['mask1'][DTInfo['mask1']>1] = 0
        DTInfo['max1'] = nmax_1
        DTInfo['min1'] = nmin_1

        # select the 70 slices (70 is the smallest)
        DTInfo['normed_data2'] = np.transpose(normed_data_2, [2,0,1]) # get the nslice*96*96
        DTInfo['bvalue2'] = mat['bvalue'][0,-self.slices:] # :nslice
        DTInfo['dirs4PC2'] = mat['dirs4PC'][-self.slices:,:] # (:nslice, 3)
        # for all masks, make the gray with 1 and others 0.
        DTInfo['mask2'] = np.transpose(mat['mask'][80:(96+80),...,-self.slices:], [2,0,1]) # (nslice*96*96)
        DTInfo['mask2'][DTInfo['mask2']>1] = 0
        DTInfo['max2'] = nmax_2
        DTInfo['min2'] = nmin_2

        # for all masks, make the gray with 1 and others 0.
        DTInfo['template_mask'] = mat['template_mask'][:,80:(96+80)]
        DTInfo['template_mask'][DTInfo['template_mask']>1] = 0
        if rotate == 1:
            # transpose the data if the orientation is not correct to match different bvalues.
            DTInfo['normed_data1'] = np.transpose(DTInfo['normed_data1'], [0,2,1])
            DTInfo['mask1'] = np.transpose(DTInfo['mask1'], [0,2,1])
            DTInfo['template_mask'] = np.transpose(DTInfo['template_mask'], [0,2,1])
            DTInfo['normed_data2'] = np.transpose(DTInfo['normed_data2'], [0,2,1])
            DTInfo['mask2'] = np.transpose(DTInfo['mask2'], [0,2,1])
        DTInfo['subject'] = self.denoised_cases[index]
        DTInfo['idx2d'] = idx2d
        DTInfo['o_slices'] = origin_slices
        return DTInfo