import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F
from scipy.io import loadmat
import os
from os import listdir
import pandas as pd
from skimage import io
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle
import operator
import pdb
import random

def create_files_new(load_path, bad_dates, T_length=10, proceed=1):
    '''
    Order and group time windows.
    
    Input:
    load_path: the path to load time windows
    bad_dates: list of dates we choose not to use due to irregularities
    T_length: the number of consecutive time windows we group together
    proceed: the number that controls the overlap between two groups
             the number of overlapping time windows between two consecutive groups is T_length minus proceed
             
    Output:
    move_data: list of lists containing movement group information
    sleep_data: list of lists containing sleep group information      
    '''
    
    sleep_files = os.listdir(load_path+'sleep/')
    move_files = os.listdir(load_path+'move/')
    all_files = sleep_files+move_files
    
    dic = {}
    for f in all_files:     
        # extract information contained in names of the time windows
        mvmt_type = f.split('_')[-1].split('.')[0]
        date = f.split('_')[0]
        rec = f.split('_')[1].split('_')[0]
        time = float(f.split('_')[3][4:])
        if date in bad_dates:
            continue
        if mvmt_type == 'sleep':
            label = 1
        else:
            label = 0
        # store data in dictionary with keys date and rec
        if date in dic:
            if rec in dic[date]:
                dic[date][rec].append([f, label, mvmt_type, date, rec, time])
            else:
                dic[date][rec] = [[f, label, mvmt_type, date, rec, time]]
        else:
            dic[date] = {rec: [[f, label, mvmt_type, date, rec, time]]}
        
    # sort data by date, rec and time 
    for d in dic:
        for r in dic[d]:
            dic[d][r] = sorted(dic[d][r], key=operator.itemgetter(3, 4, 5))
    
    # call helper function to group
    move_data, sleep_data = [], []
    for d in dic:
        for r in dic[d]:
            sleep_grouped, move_grouped = create_files_new_helper(dic[d][r], T_length=T_length, proceed=proceed)
            sleep_data.append(sleep_grouped)
            move_data.append(move_grouped)
    
    return move_data, sleep_data

def create_files_new_helper(L, T_length, proceed):
    '''
    Help group data.
    
    Input:
    L: list of time windows for a given record of a given date
    T_length: the number of consecutive time windows we group together
    proceed: the number that controls the overlap between two groups
             the number of overlapping time windows between two consecutive groups is T_length minus proceed
             
    Output:
    L_new_sleep: list of groups of sleep windows
    L_new_move: list of groups of movement windows
    '''
    L_labels = np.array([L[i][1] for i in range(len(L))])
    L_times = np.array([L[i][-1] for i in range(len(L))])
    
    L_new_sleep, L_new_move = [], []
    start = 0
    while start <= len(L)-T_length:
        end = start + T_length
        # increase start point if the next T_length windows are not consecutive
        if sum(L_times[start+1:end]-L_times[start:end-1]-time_window) != 0:
            start += 1
            continue
            
        # decide if the group belongs to movement or sleep
        if sum(L_labels[start:end]) == T_length:
            L_new_sleep.append(L[start:end])
        elif sum(L_labels[start:end]) == 0:
            L_new_move.append(L[start:end])
        start += proceed
    return L_new_sleep, L_new_move

def upsample(train_files):
    '''
    Upsample to have the same amount of movement and sleep groups
    
    Input:
    train_files: list of groups of time windows
    
    Output:
    train_files: new list with balanced labels
    '''
    train_sleep = [i for i in train_files if i[0][1] == 1]
    train_move = [i for i in train_files if i[0][1] == 0]
    diff = abs(len(train_sleep)-len(train_move))
    train_new = []
    d = 0
    while d < diff:
        # see if we want to upsample movement or sleep
        # most of cases we want to upsample movement
        if len(train_sleep) > len(train_move):
            ind = random.randint(0, len(train_move)-1)
            x = train_move[ind]
            d += 1
        else:
            ind = random.randint(0, len(train_sleep)-1)
            x = train_sleep[ind]
            d += 1
        train_new.append(x)   
    train_files = train_sleep+train_move+train_new
    return train_files

class SpectrogramDatasetAtt(Dataset):
    def __init__(self, files, load_path, T_length, all_label=False, CH=None):
        '''
        Input:
        files: list of groups of time windows
        load_path: the path to load time windows
        T_length: the number of consecutive time windows we group together
        all_label: whether we want all labels or only label for the middle time window
        CH: specify which channel we want to extract if decide not to use all (CH=None)        
        '''
        self.CH = CH
        self.files = files
        self.load_path = load_path
        self.T_length = T_length
        self.all_label = all_label
  
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        group = self.files[idx]
        specs, labels, dates, recs, times = [], [], [], [], []
        for i in range(len(group)):
            f, label, mvmt_type, date, rec, time = group[i]
            spec = torch.from_numpy(np.load(self.load_path+mvmt_type+'/'+f))
            if self.CH is not None:
                spec = torch.transpose(spec[self.CH,:,:].unsqueeze(0), 2, 1)
            else:
                spec = torch.transpose(spec, 2, 1)
            specs.append(spec)
            labels.append(torch.Tensor([label]))
            dates.append(date)
            recs.append(rec)
            times.append(time)
            
            # save information for middle time windows
            if i == (self.T_length-1)/2:
                label_mid = torch.Tensor([label])
                date_mid = date
                rec_mid = rec
                time_mid = time
                
        if self.all_label:
            return specs, labels, dates, recs, times
        else:
            return specs, label_mid, date_mid, rec_mid, time_mid
        
def position_encoding_init(n_position, emb_dim):
    '''
    Initialize the positional encoder from BERT paper
    
    Input:
    n_position: the number of words in a sentence
    emb_dim: the length of representations for each word
    
    Output:
    initialized positional embedding with dimensions (n_position, emb_dim)
    '''
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class Attention(nn.Module):
    # Implementation of Attention is all you need
    # Does not work in our case
    # Leave here in case for future reference
    def __init__(self, att_dim, T_length):
        super(Attention, self).__init__()
        
        # Positional Encoding
        self.position_enc = nn.Embedding(T_length, 4*25*2)
        self.position_enc.weight.data = position_encoding_init(T_length, 4*25*2)

        # Attention
        self.query = nn.Linear(4*25*2, att_dim)
        self.key = nn.Linear(4*25*2, att_dim)
        self.value = nn.Linear(4*25*2, att_dim)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(4*25*2+att_dim, 1)
        
        self.T_length = T_length
        
    def forward(self, x, x_pos):
        x = x.float()
        x += self.position_enc(x_pos)

        x_query, x_key, x_value = self.query(x), self.key(x), self.value(x)
        energy =  torch.bmm(x_query, x_key.permute(0, 2, 1))
        attention = self.softmax(energy)
        out = torch.bmm(attention, x_value)
        x_concat = torch.cat((x[:, int((self.T_length-1)/2), :], out[:, int((self.T_length-1)/2), :]), 1)
        x_output = self.fc1(x_concat)
        x_output = torch.sigmoid(x_output)
        
        return x_output
    
def get_accuracy(model_CNN, model_Att, loader, device='cuda'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels, _, _, _ in loader:
            labels = torch.stack(labels).transpose(1,0)
            labels = labels.to(device).float()
            CNN_outputs = torch.zeros(data[0].shape[0], len(data), att_dim, device=device)
            for t in range(T_length):
                data_t = data[t].to(device).float()
                CNN_outputs[:, t, :] = model_CNN(data_t) # batch size, number of time windows, attention dimension
            pos = torch.LongTensor([list(range(T_length))]).to(device)
            Att_outputs = model_Att(CNN_outputs, pos)
            predictions = (Att_outputs > 0.5) * 1.0
            predictions = predictions.flatten().detach().cpu().numpy()
            labels = labels.flatten().cpu().numpy()
            total += len(labels)
            correct += (predictions == labels).sum()
            
    accuracy = correct / total
    return accuracy

def train(model_CNN, model_Att, optimizer_CNN, optimizer_Att, criterion, loader, T_length, device='cuda'):
    model_CNN.train()
    model_Att.train()
    epoch_losses = 0
    epoch_lens = 0
    
    for batch_idx, (data, labels, _, _, _) in enumerate(loader):
        labels = torch.stack(labels).transpose(1,0)
        labels = labels.to(device).float()
        # Initialize and fill in CNN outputs
        CNN_outputs = torch.zeros(data[0].shape[0], len(data), att_dim, device=device)
        for t in range(T_length):
            data_t = data[t].to(device).float()
            CNN_outputs[:, t, :] = model_CNN(data_t)
            
        # Initialized positional embedding and collect outputs from self-attention
        pos = torch.LongTensor([list(range(T_length))]).to(device)
        Att_outputs = model_Att(CNN_outputs, pos)
        loss = criterion(Att_outputs, labels)
        epoch_losses += loss
        epoch_lens += 1

        optimizer_CNN.zero_grad()
        optimizer_Att.zero_grad()
        loss.backward() # Will back propogation work correctly?
        optimizer_CNN.step()
        optimizer_Att.step()
    
    epoch_accs = get_accuracy(model_CNN, model_Att, loader, device=device)
    return epoch_losses/epoch_lens, epoch_accs

def evaluate(model_CNN, model_Att, optimizer_CNN, optimizer_Att, criterion, loader, T_length, device='cuda'):
    model_CNN.eval()
    model_Att.eval()
    epoch_losses = 0
    epoch_lens = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels, _, _, _) in enumerate(loader):
            labels = torch.stack(labels).transpose(1,0)
            labels = labels.to(device).float()
            CNN_outputs = torch.zeros(data[0].shape[0], len(data), att_dim, device=device)
            for t in range(T_length):
                data_t = data[t].to(device).float()
                CNN_outputs[:, t, :] = model_CNN(data_t)
            pos = torch.LongTensor([list(range(T_length))]).to(device)
            Att_outputs = model_Att(CNN_outputs, pos)
            loss = criterion(Att_outputs, labels)
            epoch_losses += loss
            epoch_lens += 1
    
    epoch_accs = get_accuracy(model_CNN, model_Att, loader, device=device)
    return epoch_losses/epoch_lens, epoch_accs
    