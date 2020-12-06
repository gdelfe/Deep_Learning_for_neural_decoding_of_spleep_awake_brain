"""
Shared data loader object for training on
 movement vs non-movement neural activity classification.



TODO: Monkey-wise labels, meaning also say whether or not this spec is from Goose or Jester. 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.functional import relu
from scipy.io import loadmat
import os
from os import listdir
import pandas as pd
from skimage import io
from PIL import Image
from scipy.io import loadmat



# v4 = Krishan approved labels, all 62 channels
# v3 = Krishan approved labels, one channel
version = 'v3'



#selected subset of dates. 
val_dates = ['1803238','180329','180330','180331','180410','180411','180412', '180413']

class SpectrogramDataset(Dataset):
    def __init__(self, mode='train' ,version='v4',val_dates=val_dates):
        self.version = version
        self.movement_files = os.listdir('/mnt/pesaranlab/People/Capstone_students/Noah/data'+version+'/move/')
        self.sleeping_files = os.listdir('/mnt/pesaranlab/People/Capstone_students/Noah/data'+version+'/sleep/')
        all_files = self.sleeping_files + self.movement_files
        if mode == 'train':
            self.all_files = [f for f in all_files if f.split('_')[0] not in val_dates]
        elif mode == 'valid':
            self.all_files = [f for f in all_files if f.split('_')[0] in val_dates]

            
        # clean
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
       # labe
       # spec = torch.from_numpy(np.load('data/'self.all_files[idx])).unsqueeze(0)
        mvmt_type  = self.all_files[idx].split('_')[-1].split('.')[0] #check this out
        date = self.all_files[idx].split('_')[0]
        rec = self.all_files[idx].split('_')[1].split('_')[0]
        spec = torch.from_numpy(np.load('/mnt/pesaranlab/People/Capstone_students/Noah/data'+self.version+'/'+ mvmt_type +'/' +self.all_files[idx])) 
        if mvmt_type == 'move':
            label = torch.Tensor([0])
        elif mvmt_type == 'sleep':
            label = torch.Tensor([1])
        else:
            label = torch.Tensor([-1])
        return  spec.resize(1,spec.shape[0],10,100) , label, date, rec




def create_dataloaders(version='v4',batch_size=32):

    train_dataset = SpectrogramDataset(mode='train',version=version)
    valid_dataset = SpectrogramDataset(mode='valid',version=version)


    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)


    return train_loader, val_loader
