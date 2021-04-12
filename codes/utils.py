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
import random
import pdb


class SpectrogramDataset(Dataset):
    def __init__(self, val_dates, test_dates, mode='train', version='_Goose_1st', CH=None, upsample=False):
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


def create_dataloaders(val_dates, test_dates, version='v5', batch_size=32, CH=None, upsample=False):

    train_dataset = SpectrogramDataset(val_dates=val_dates, test_dates=test_dates, mode='train', version=version, CH=CH, upsample=upsample)
    valid_dataset = SpectrogramDataset(val_dates=val_dates, test_dates=test_dates, mode='valid', version=version, CH=CH, upsample=upsample)
    test_dataset = SpectrogramDataset(val_dates=val_dates, test_dates=test_dates, mode='test', version=version, CH=CH, upsample=upsample)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader


class GLM(nn.Module):
    def __init__(self, input_dim=100*10*62, output_dim=1):
        super(GLM, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim) # by default, add an intercept

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, -1]).float()
        outputs = self.linear(x)
        return outputs

    
def get_pred(outputs, model_type='LR'):
    if model_type == 'SVM':
        pred = (outputs > 0) * 1.0
    else:
        pred = (torch.sigmoid(outputs) > 0.5) * 1.0
    return pred
    
    
def get_accuracy(model, loader, model_type='LR', collect_result=False, device='cuda'):
    correct = 0
    total = 0
    preds, preds_probs, labs, cases_wrong = [], [], [], []
    with torch.no_grad():
        for data, labels, dates, recs, times in loader:
            data = data.to(device)
            labels = labels.to(device).float()
            outputs = model(data)
            predictions = get_pred(outputs, model_type=model_type)
            outputs = outputs.flatten().detach().cpu().numpy()
            predictions = predictions.flatten().detach().cpu().numpy()
            labels[labels == -1] = 0
            labels = labels.flatten().cpu().numpy()
            total += len(labels)
            correct += (predictions == labels).sum()
            
            if collect_result:
                preds.append(predictions)
                preds_probs.append(outputs)
                labs.append(labels)
                
                indices_wrong = np.argwhere(np.array(predictions != labels)).flatten()
                if len(indices_wrong) == 0:
                    continue
                dates_wrong = np.array(dates)[indices_wrong]
                recs_wrong = np.array(recs)[indices_wrong]
                times_wrong = np.array(times)[indices_wrong]
                labels_wrong = np.array(labels)[indices_wrong]
                data_wrong = np.array(data.cpu().numpy())[indices_wrong]
                cases_wrong.extend([[dates_wrong[i], 
                                     recs_wrong[i], 
                                     times_wrong[i], 
                                     labels_wrong[i], 
                                     data_wrong[i]] for i in range(len(dates_wrong))])
    accuracy = correct / total
    if collect_result:
        return accuracy, preds, preds_probs, labs, cases_wrong
    return accuracy


def get_loss(model, labels, outputs, alpha=0, loss_type='bce', reg_type='none', reduction='mean'):
    if loss_type == 'hinge':
        labels[labels == 0] = -1
        if reduction == 'mean':
            loss = torch.mean(torch.clamp(1 - labels*outputs, min=0))
        elif reduction == 'sum':
            loss = torch.sum(torch.clamp(1 - labels*outputs, min=0))
    elif loss_type == 'bce':
        if reduction == 'mean':
            criterion = nn.BCELoss(reduction='mean')
        elif reduction == 'sum':
            criterion = nn.BCELoss(reduction='sum')
        loss = criterion(torch.sigmoid(outputs), labels)
    if reg_type != 'none':
        weights = model.linear.weight.view(-1, 100, 10)
    if reg_type == 'l2':
        loss += alpha * weights.norm(2)
    elif reg_type == 'finite_diff':
        diff_h = (weights[:, :, 1:] - weights[:, :, :-1]).norm(2)
        diff_v = (weights[:, 1:, :] - weights[:, :-1, :]).norm(2)
        loss += alpha * (diff_h + diff_v)    
    return loss


def train(model, optimizer, loader, alpha, model_type='LR', loss_type='bce', reg_type=None, collect_result=False, device='cuda'):
    model.train()
    batch_losses = 0
    batch_lengths = 0
    
    for batch_idx, (data, labels, _, _, _) in enumerate(loader):
        data = data.to(device)
        labels = labels.to(device).float()
        
        if (torch.isinf(data).any()) or (torch.isnan(data).any()):
            continue
        
        outputs = model(data)
        outputs = outputs.reshape(outputs.shape[0],-1)
        loss = get_loss(model, labels, outputs, alpha=alpha, loss_type=loss_type, reg_type=reg_type, reduction='sum')
        batch_losses += loss
        batch_lengths += labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss = batch_losses/batch_lengths
    
    if collect_result:
        acc, preds, preds_probs, labs, cases_wrong = get_accuracy(model, loader, model_type=model_type, collect_result=True, device=device)
        return epoch_loss, acc, preds, preds_probs, labs, cases_wrong
    else:
        acc = get_accuracy(model, loader, model_type=model_type, collect_result=False, device=device)
        return epoch_loss, acc

def evaluate(model, optimizer, loader, alpha, model_type='LR', loss_type='bce', reg_type=None, collect_result=False, device='cuda'):
    model.eval()
    batch_losses = 0
    batch_lengths = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels, dates, recs, times) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device).float()
            
            if (torch.isinf(data).any()) or (torch.isnan(data).any()):
                continue
            
            outputs = model(data)
            outputs = outputs.reshape(outputs.shape[0],-1)
            loss = get_loss(model, labels, outputs, alpha=alpha, loss_type=loss_type, reg_type=reg_type, reduction='sum')
            batch_losses += loss
            batch_lengths += labels.shape[0]
        
    epoch_loss = batch_losses/batch_lengths 
    
    if collect_result:
        acc, preds, preds_probs, labs, cases_wrong = get_accuracy(model, loader, model_type=model_type, collect_result=True, device=device)
        return epoch_loss, acc, preds, preds_probs, labs, cases_wrong
    else:
        acc = get_accuracy(model, loader, model_type=model_type, collect_result=False, device=device)
        return epoch_loss, acc


def plot_loss_acc(training_losses, val_losses, training_acc, validation_acc, model_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.title(model_name, fontsize = 15)
    plt.plot(training_losses, linewidth = 1.5, label = 'train')
    plt.plot(val_losses, linewidth = 1.5, label = 'valid')
    plt.xlabel("Epoch",fontsize = 15)
    plt.ylabel("Loss", fontsize = 15)
    plt.legend()
    plt.subplot(1,2,2)
    plt.title(model_name, fontsize = 15)
    plt.plot(training_acc, linewidth = 1.5, label = 'train')
    plt.plot(validation_acc, linewidth = 1.5, label = 'valid')
    plt.xlabel("Epoch",fontsize = 15)
    plt.ylabel("Accuracy", fontsize = 15)
    plt.legend()
    plt.show()
    