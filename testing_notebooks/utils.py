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


#selected subset of dates. 
val_dates = ['180412','180413','180414']


class SpectrogramDataset(Dataset):
    def __init__(self, mode='train', version='v3', val_dates=val_dates, CH=None):
        self.CH = CH
        self.version = version
        self.movement_files = os.listdir('/mnt/pesaranlab/People/Capstone_students/Noah/data'+version+'/move/')
        self.sleeping_files = os.listdir('/mnt/pesaranlab/People/Capstone_students/Noah/data'+version+'/sleep/')
        all_files = self.sleeping_files + self.movement_files
        if mode == 'train':
            self.all_files = [f for f in all_files if f.split('_')[0] not in val_dates]
        elif mode == 'valid':
            self.all_files = [f for f in all_files if f.split('_')[0] in val_dates]
  
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        mvmt_type  = self.all_files[idx].split('_')[-1].split('.')[0] #check this out
        date = self.all_files[idx].split('_')[0]
        rec = self.all_files[idx].split('_')[1].split('_')[0]
        path = '/mnt/pesaranlab/People/Capstone_students/Noah/'
        spec = torch.from_numpy(np.load(path+'data'+self.version+'/'+ mvmt_type +'/' +self.all_files[idx])) 
        if mvmt_type == 'move':
            label = torch.Tensor([0])
        elif mvmt_type == 'sleep':
            label = torch.Tensor([1])
        else:
            label = torch.Tensor([-1])
            
        if self.CH is not None:
            return torch.transpose(spec[self.CH,:,:].unsqueeze(0), 2, 1), label, date, rec
        else:
            return torch.transpose(spec, 2, 1), label, date, rec


def create_dataloaders(version='v5', batch_size=32, CH=None):

    train_dataset = SpectrogramDataset(mode='train',version=version, CH=CH)
    valid_dataset = SpectrogramDataset(mode='valid',version=version, CH=CH)


    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)


    return train_loader, val_loader


class LogReg(nn.Module):
    def __init__(self, input_dim=100*10*62, output_dim=1):
        super(LogReg, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, -1]).float()
        outputs = torch.sigmoid(self.linear(x))
        
        return outputs
    
def get_accuracy(model, loader, device='cuda'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels, dates, recs in loader:
            data = data.to(device)
            labels = labels.to(device).float()
            outputs = model(data)
            predictions = []
            for o in outputs:
                if o.item() > .5:
                    predictions.append(1)
                else:
                    predictions.append(0)
            predictions = np.array(predictions)
            total += labels.size(0)
            correct += (predictions.flatten() == labels.flatten().cpu().numpy()).sum().item()
            
    return correct / total

def train(model, criterion, optimizer, train_loader, epoch, alpha, device='cuda'):
    model.train()
    batch_losses = []
    
    preds = []
    preds_probs = []
    labs = []
    
    correct = 0
    total = 0
    
    for batch_idx, (data, labels, _, _) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device).float()
        
        if (torch.isinf(data).any()) or (torch.isnan(data).any()):
            continue
        
        outputs = model(data)
        loss = criterion(outputs.reshape(outputs.shape[0],-1), labels)
        W = model.linear.weight.view(62, 100, 10)
        diff_h = (W[:, :, 1:] - W[:, :, :-1]).norm(2)
        diff_v = (W[:, 1:, :] - W[:, :-1, :]).norm(2)
        loss += alpha*(diff_h + diff_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss)
        
        predictions = []
        for o in outputs:
            if o.item() > .5:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions)
        total += labels.size(0)
        correct += (predictions.flatten() == labels.flatten().cpu().numpy()).sum().item()
        
        preds.append(predictions.flatten())
        preds_probs.append(outputs.flatten().cpu().detach().numpy())
        labs.append(labels.flatten().cpu().numpy())
        
    epoch_loss = sum(batch_losses)/len(batch_losses)
    accuracy = correct / total
    # acc = get_accuracy(train_loader) 
    
    return epoch_loss, accuracy, preds, preds_probs, labs

def test(model, criterion, optimizer, val_loader, device='cuda'):
    model.eval()
    batch_losses = []
    preds = []
    preds_probs = []
    labs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels, _, _) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device).float()
            
            if (torch.isinf(data).any()) or (torch.isnan(data).any()):
                continue
            
            outputs = model(data)
            loss = criterion(outputs.reshape(outputs.shape[0],-1), labels)
            
            batch_losses.append(loss)
            
            predictions = []
            for o in outputs:
                if o.item() > .5:
                    predictions.append(1)
                else:
                    predictions.append(0)
            predictions = np.array(predictions)
            total += labels.size(0)
            correct += (predictions.flatten() == labels.flatten().cpu().numpy()).sum().item()
        
            preds.append(predictions.flatten())
            preds_probs.append(outputs.flatten().cpu().detach().numpy())
            labs.append(labels.flatten().cpu().numpy())
        
        epoch_loss = sum(batch_losses)/len(batch_losses)
        
    accuracy = correct/total
    # acc = get_accuracy(loader)
    
    return epoch_loss, accuracy, preds, preds_probs, labs