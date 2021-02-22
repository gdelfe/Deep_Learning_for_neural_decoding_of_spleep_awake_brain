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
val_dates = ['180328','180329']
test_dates = ['180330','180331']

class SpectrogramDataset(Dataset):
    def __init__(self, mode='train', version='_Goose_1st', val_dates=val_dates, CH=None, upsample=False):
        self.CH = CH
        self.version = version
        self.movement_files = os.listdir('/mnt/pesaranlab/People/Capstone_students/Yue/data/data'+version+'/move/')
        self.sleeping_files = os.listdir('/mnt/pesaranlab/People/Capstone_students/Yue/data/data'+version+'/sleep/')
        if upsample:
            diff = len(self.sleeping_files)-len(self.movement_files)
            try:
                d = 0
                while d < diff:
                    ind = random.randint(0, len(self.movement_files)-1)
                    x = self.movement_files[ind]
                    x_date = x.split('_')[0]
                    if x_date not in val_dates+test_dates:
                        self.movement_files.append(x)
                        d += 1
            except ValueError:
                print('Movoment instance more than sleep instances!')
        all_files = self.sleeping_files + self.movement_files
        if mode == 'train':
            self.all_files = [f for f in all_files if f.split('_')[0] not in val_dates+test_dates]
        elif mode == 'valid':
            self.all_files = [f for f in all_files if f.split('_')[0] in val_dates]
        elif mode == 'test':
            self.all_files = [f for f in all_files if f.split('_')[0] in test_dates]
  
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        mvmt_type  = self.all_files[idx].split('_')[-1].split('.')[0]
        date = self.all_files[idx].split('_')[0]
        rec = self.all_files[idx].split('_')[1].split('_')[0]
        time = float(self.all_files[idx].split('_')[3][4:])
        path = '/mnt/pesaranlab/People/Capstone_students/Yue/data/'
        spec = torch.from_numpy(np.load(path+'data'+self.version+'/'+ mvmt_type +'/' +self.all_files[idx])) 
        if mvmt_type == 'move':
            label = torch.Tensor([0])
        elif mvmt_type == 'sleep':
            label = torch.Tensor([1])
        else:
            label = torch.Tensor([-1])
            
        if self.CH is not None:
            return torch.transpose(spec[self.CH,:,:].unsqueeze(0), 2, 1), label, date, rec, time
        else:
            return torch.transpose(spec, 2, 1), label, date, rec, time


def create_dataloaders(version='v5', batch_size=32, CH=None):

    train_dataset = SpectrogramDataset(mode='train',version=version, CH=CH)
    valid_dataset = SpectrogramDataset(mode='valid',version=version, CH=CH)
    test_dataset = SpectrogramDataset(mode='test',version=version, CH=CH)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader


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
        for data, labels, dates, recs, times in loader:
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
    
    for batch_idx, (data, labels, _, _, _) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device).float()
        
        if (torch.isinf(data).any()) or (torch.isnan(data).any()):
            continue
        
        outputs = model(data)
        loss = criterion(outputs.reshape(outputs.shape[0],-1), labels)
        try:
            W = model.linear.weight.view(62, 100, 10)
        except:
            W = model.linear.weight.view(1, 100, 10)
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

def test(model, criterion, optimizer, val_loader, device='cuda', mode='valid'):
    model.eval()
    batch_losses = []
    preds = []
    preds_probs = []
    labs = []
    correct = 0
    total = 0
    cases_wrong = []
    
    with torch.no_grad():
        for batch_idx, (data, labels, dates, recs, times) in enumerate(val_loader):
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
            
            if mode == 'test':
                indices_wrong = np.argwhere(np.array(predictions.flatten() != labels.flatten().cpu().numpy())).flatten()
                if len(indices_wrong) == 0:
                    continue
                dates_wrong = np.array(dates)[indices_wrong]
                recs_wrong = np.array(recs)[indices_wrong]
                times_wrong = np.array(times)[indices_wrong]
                labels_wrong = np.array(labels.flatten().cpu().numpy())[indices_wrong]
                data_wrong = np.array(data.cpu().numpy())[indices_wrong]
                cases_wrong.extend([[dates_wrong[i], 
                                     recs_wrong[i], 
                                     times_wrong[i], 
                                     labels_wrong[i], 
                                     data_wrong[i]] for i in range(len(dates_wrong))])
        
        epoch_loss = sum(batch_losses)/len(batch_losses)
        
    accuracy = correct/total
    # acc = get_accuracy(loader)
    
    if mode == 'test':
        return epoch_loss, accuracy, preds, preds_probs, labs, cases_wrong
    return epoch_loss, accuracy, preds, preds_probs, labs