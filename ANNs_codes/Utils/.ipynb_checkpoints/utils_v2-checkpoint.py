#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:47:31 2021

@author: bijanadmin
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
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pdb



class SpectrogramDatasetNew(Dataset):
    def __init__(self, files, load_path, CH=None):
        self.CH = CH
        self.files = files
        self.load_path = load_path
  
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f, label, mvmt_type, date, rec, time = self.files[idx]     # get file_name, 0/1, sleep/move, date, recording, time 
        spec = torch.from_numpy(np.load(self.load_path+mvmt_type+'/'+f)) # load spectrogram for the specific file
        if self.CH is not None: # if only one channel
            return torch.transpose(spec[self.CH,:,:].unsqueeze(0), 2, 1), torch.Tensor([label]), date, rec, time
        else: # if all the channels
            return torch.transpose(spec, 2, 1), torch.Tensor([label]), date, rec, time
   
        
#    Create training files for both movement and sleep, upsample movement data

def create_files(load_path, val_dates, test_dates, bad_dates):
    train_files, val_files, test_files = [], [], []
    sleep_files = os.listdir(load_path+'sleep/')
    move_files = os.listdir(load_path+'/move/')

    # create list of sleep and movement files containing:
    # name of file, movement (0/1), movement (sleep/move), date, rec, time)
    all_files = sleep_files+move_files
    for f in all_files:
        mvmt_type = f.split('_')[-1].split('.')[0]
        if mvmt_type == 'sleep':
            label = 1
        elif mvmt_type == 'move':
            label = 0
        date = f.split('_')[0]
        rec = f.split('_')[1].split('_')[0]
        time = float(f.split('_')[3][4:])
        if date not in val_dates+test_dates+bad_dates:
            train_files.append([f, label, mvmt_type, date, rec, time])
        elif date in val_dates:
            val_files.append([f, label, mvmt_type, date, rec, time])
        elif date in test_dates:
            test_files.append([f, label, mvmt_type, date, rec, time])
            
    # count sleep and move occurrencies in training set only 
    train_sleep = [i for i in train_files if i[1] == 1]    # sleep occurrencies
    train_move = [i for i in train_files if i[1] == 0]     # move occurrencies 
    diff = len(train_sleep)-len(train_move)
    # upsample movement occurrencies 
    try:
        d = 0
        while d < diff:
            ind = random.randint(0, len(train_move)-1)
            x = train_move[ind]
            train_move.append(x)
            d += 1
    except ValueError:
        print('Movoment instance more than sleep instances!')
    train_files = train_sleep+train_move   # total instances of training files 
            
    return train_files, val_files, test_files
    
# Data Loader for train, validation, and test
def create_dataloaders(train_files, val_files, test_files, load_path, batch_size=32, CH=None):
    train_dataset = SpectrogramDatasetNew(files=train_files, load_path=load_path, CH=CH)
    valid_dataset = SpectrogramDatasetNew(files=val_files, load_path=load_path, CH=CH)
    test_dataset = SpectrogramDatasetNew(files=test_files, load_path=load_path, CH=CH)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, val_loader, test_loader

def test_imbalance(load_path, val_dates, test_dates, bad_dates, test_train=False):
    train_files, val_files, test_files = create_files(load_path, val_dates, test_dates, bad_dates)
    train_loader, val_loader, test_loader = create_dataloaders(train_files, val_files, test_files, load_path, batch_size=256)
    val_labels, test_labels = [], []
    for _, labels, _, _, _ in val_loader:
        val_labels.extend(list(labels.flatten().numpy()))
    for _, labels, _, _, _ in test_loader:
        test_labels.extend(list(labels.flatten().numpy()))
    print('val dates: {}, test dates: {}'.format(val_dates, test_dates))
    print('val instances: {}, val imbalance: {}'.format(len(val_labels), np.mean(val_labels)))
    print('test instances: {}, test imbalance: {}'.format(len(test_labels), np.mean(test_labels)))
    if test_train:
        train_labels = []
        for _, labels, _, _, _ in train_loader:
            train_labels.extend(list(labels.flatten().numpy()))
        print('train instances: {}, train imbalance: {}'.format(len(train_labels), np.mean(train_labels)))


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
    preds, preds_probs, labs, dates_all, recs_all, times_all, cases_wrong = [], [], [], [], [], [], []
    with torch.no_grad():
        for data, labels, dates, recs, times in loader:
            data = data.to(device).float()
            labels = labels.to(device).float()
            outputs, conv1, conv2 = model(data)
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
                dates_all.append(dates)
                recs_all.append(recs)
                times_all.append(times)
                
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
        return accuracy, preds, preds_probs, labs, dates_all, recs_all, times_all, cases_wrong
    return accuracy


def get_loss(model, labels, outputs, alpha=0, timewindow=10, loss_type='bce', reg_type='none', reduction='mean'):
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
    # if reg_type != 'none':
        # weights = model.linear.weight.view(-1, 100, timewindow)
    if reg_type == 'l2':
        loss += alpha * weights.norm(2)
    elif reg_type == 'finite_diff':
        diff_h = (weights[:, :, 1:] - weights[:, :, :-1]).norm(2)
        diff_v = (weights[:, 1:, :] - weights[:, :-1, :]).norm(2)
        loss += alpha * (diff_h + diff_v)    
    return loss





def train(model, optimizer, criterion, loader, alpha, timewindow=10, model_type='LR', loss_type='bce', reg_type=None, collect_result=False, device='cuda'):
    model.train()
    batch_losses = 0
    batch_lengths = 0
    
    for batch_idx, (data, labels, _, _, _) in enumerate(loader):
        data = data.to(device).float()
        labels = labels.to(device).float()
        
        outputs, conv1, conv2 = model(data)
        outputs = outputs.reshape(outputs.shape[0],-1)
        # loss = get_loss(model, labels, outputs, alpha=alpha, timewindow=timewindow, loss_type=loss_type, reg_type=reg_type, reduction='sum')
        loss = criterion(torch.sigmoid(outputs), labels)
        batch_losses += loss
        batch_lengths += labels.shape[0]
        loss /= labels.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss = batch_losses/batch_lengths
    
    if collect_result:
        acc, preds, preds_probs, labs, cases_wrong = get_accuracy(model, loader, model_type=model_type, collect_result=True, device=device)
        return epoch_loss, acc, preds, preds_probs, labs, cases_wrong
    else:
        acc = get_accuracy(model, loader, model_type=model_type, collect_result=False, device=device)
        return epoch_loss, acc, conv1, conv2

def evaluate(model, optimizer, criterion, loader, alpha, timewindow=10, model_type='LR', loss_type='bce', reg_type=None, collect_result=False, device='cuda'):
    model.eval()
    batch_losses = 0
    batch_lengths = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels, dates, recs, times) in enumerate(loader):
            data = data.to(device).float()
            labels = labels.to(device).float()

            outputs, conv1, conv2 = model(data)
            outputs = outputs.reshape(outputs.shape[0],-1)
            # loss = get_loss(model, labels, outputs, alpha=alpha, timewindow=timewindow, loss_type=loss_type, reg_type=reg_type, reduction='sum')
            loss = criterion(torch.sigmoid(outputs), labels)
            batch_losses += loss
            batch_lengths += labels.shape[0]
        
    epoch_loss = batch_losses/batch_lengths 
    
    if collect_result:
        acc, preds, preds_probs, labs, dates_all, recs_all, times_all, cases_wrong = get_accuracy(model, loader, model_type=model_type, collect_result=True, device=device)
        return epoch_loss, acc, preds, preds_probs, labs, dates_all, recs_all, times_all, cases_wrong
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
    

def plot_weight_glm(device, path, model_type, CH, loss_type, reg_type, alpha, best_epoch, timewindow=10):
    if CH == 'all':
        model = GLM(input_dim=100*timewindow*62).to(device)
        model.load_state_dict(torch.load('{}/{}_CH{}_LOSS{}_REG{}{}_TW{}_EPOCH{}.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, timewindow,best_epoch)))
        weights = model.linear.weight.view(62, 100, timewindow)
        plt.figure(figsize=(15,8))
        for i in range(62):
            weights_sub = weights[i].detach().cpu()
            plt.subplot(2,31,i+1)
            if i==0 or i==31:
                plt.yticks(ticks=[0, 20, 40, 60, 80, 99], labels=[round(np.logspace(0, 2.45, 100)[i]) for i in [0, 20, 40, 60, 80, 99]])
                plt.xticks(ticks=[0, timewindow-1], labels=[1, timewindow])
            else:
                plt.axis('off')
            plt.imshow(weights_sub)
            plt.title(str(i+1))
        plt.show()
    else:
        model = GLM(input_dim=100*timewindow).to(device)
        model.load_state_dict(torch.load('{}/{}_CH{}_LOSS{}_REG{}{}_TW{}_EPOCH{}.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, timewindow, best_epoch)))
        weights = model.linear.weight.view(100, timewindow)
        weights_sub = weights.detach().cpu()
        plt.yticks(ticks=[0, 20, 40, 60, 80, 99], labels=[round(np.logspace(0, 2.45, 100)[i]) for i in [0, 20, 40, 60, 80, 99]])
        plt.xticks(ticks=[0, timewindow-1], labels=[1, timewindow])
        plt.imshow(weights_sub)
        plt.title(str(CH))
        plt.show()
        
def plot_confusion(test_preds, test_labels,title):
    predictions_test = np.concatenate(test_preds)
    labels_test = np.concatenate(test_labels)

    cm_test = confusion_matrix(labels_test, predictions_test)
    cm_test_percent = (cm_test.T/cm_test.astype(np.float).sum(axis=1)).T
    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    sn.heatmap(cm_test, annot = True,  fmt = 'd', cmap='Blues')
    plt.title('{} Confusion Matrix'.format(title))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.subplot(1,2,2)
    sn.heatmap(cm_test_percent, annot = True, cmap='Blues')
    plt.title('{} Confusion Matrix (Rate)'.format(title))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
def plot_pred_vs_true(preds, labels, dates, recs, times, date_1='180329', rec_list=['001', '002', '003', '004', '005', '006', '007', '008', '009']):
    pred, label, date, rec, time = [], [], [], [], []
    for i in range(len(preds)):
        pred.extend(list(preds[i]))
        label.extend(list(labels[i]))
        date.extend(list(dates[i]))
        rec.extend(list(recs[i]))
        time.extend(list(times[i].numpy()))
    
    dic = {'pred': pred, 'label': label, 'date': date, 'rec': rec, 'time': time}
    df = pd.DataFrame(dic)
    
    for rec in rec_list:
        df_rec = df[(df.date == date_1) & (df.rec == rec)].sort_values(by='time')
        if not df_rec.empty:
            plt.figure(figsize=(20,6))
            plt.plot(df_rec.time, df_rec.pred, 'o', label='pred')
            plt.plot(df_rec.time, df_rec.label, '+', label='true')
            plt.legend()
            plt.xlabel('time',fontsize=15)
            plt.title('wake/sleep classification for date {} and rec {}'.format(date_1, rec), fontsize=15)
            plt.show()
        
        
def tuning(train_loader, val_loader, model, optimizer, device, num_epochs, alpha, model_type, loss_type, reg_type, CH, path, timewindow=10, verbose=False):
    training_losses, training_acc, val_losses, validation_acc = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, alpha=alpha, timewindow=timewindow, model_type=model_type, loss_type=loss_type, reg_type=reg_type, collect_result=False, device=device)
        val_loss, val_acc = evaluate(model, optimizer, criterion, val_loader, alpha=alpha, timewindow=timewindow, model_type=model_type, loss_type=loss_type, reg_type=reg_type, collect_result=False, device=device)
        training_losses.append(train_loss)
        training_acc.append(train_acc)
        val_losses.append(val_loss)
        validation_acc.append(val_acc)
        if val_loss <= min(val_losses):
            best_epoch = epoch
            print(epoch)
            print('Train loss for epoch {}: {}'.format(epoch, train_loss))
            print('Val loss for epoch {}: {}'.format(epoch, val_loss))
            torch.save(model.state_dict(), '{}/{}_CH{}_LOSS{}_REG{}{}_TW{}_EPOCH{}.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, timewindow, epoch))
        elif verbose:
            print('Train loss for epoch {}: {}'.format(epoch, train_loss))
            print('Val loss for epoch {}: {}'.format(epoch, val_loss))
        
        # so we could calculate the confusion matrix for train data when its loss reaches around minimum
        if epoch == num_epochs-1:
            torch.save(model.state_dict(), '{}/{}_CH{}_LOSS{}_REG{}{}_TW{}_EPOCH{}.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, timewindow, epoch))
            
    plot_loss_acc(training_losses, val_losses, training_acc, validation_acc, model_type)  
    return best_epoch, min(val_losses)
    