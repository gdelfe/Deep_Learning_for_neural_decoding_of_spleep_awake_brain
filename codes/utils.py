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
from sklearn.metrics import confusion_matrix
import seaborn as sn


class SpectrogramDatasetNew(Dataset):
    def __init__(self, files, load_path, CH=None):
        self.CH = CH
        self.files = files
        self.load_path = load_path
  
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f, label, mvmt_type, date, rec, time = self.files[idx]
        spec = torch.from_numpy(np.load(self.load_path+mvmt_type+'/'+f))
        if self.CH is not None:
            return torch.transpose(spec[self.CH,:,:].unsqueeze(0), 2, 1), torch.Tensor([label]), date, rec, time
        else:
            return torch.transpose(spec, 2, 1), torch.Tensor([label]), date, rec, time
        
def create_files(load_path, val_dates, test_dates):
    train_files, val_files, test_files = [], [], []
    sleep_files = os.listdir(load_path+'sleep/')
    move_files = os.listdir(load_path+'/move/')
    
    diff = len(sleep_files)-len(move_files)
    try:
        d = 0
        while d < diff:
            ind = random.randint(0, len(move_files)-1)
            x = move_files[ind]
            x_date = x.split('_')[0]
            if x_date not in val_dates+test_dates:
                move_files.append(x)
                d += 1
    except ValueError:
        print('Movoment instance more than sleep instances!')

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
        if date not in val_dates+test_dates:
            train_files.append([f, label, mvmt_type, date, rec, time])
        elif date in val_dates:
            val_files.append([f, label, mvmt_type, date, rec, time])
        elif date in test_dates:
            test_files.append([f, label, mvmt_type, date, rec, time])
            
    return train_files, val_files, test_files
    
        
def create_dataloaders(train_files, val_files, test_files, load_path, batch_size=32, CH=None):
    train_dataset = SpectrogramDatasetNew(files=train_files, load_path=load_path, CH=CH)
    valid_dataset = SpectrogramDatasetNew(files=val_files, load_path=load_path, CH=CH)
    test_dataset = SpectrogramDatasetNew(files=test_files, load_path=load_path, CH=CH)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle = False)

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
            # labels = labels.to(device).float()
            labels = labels.to(device)
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
    if reg_type != 'none':
        weights = model.linear.weight.view(-1, 100, timewindow)
    if reg_type == 'l2':
        loss += alpha * weights.norm(2)
    elif reg_type == 'finite_diff':
        diff_h = (weights[:, :, 1:] - weights[:, :, :-1]).norm(2)
        diff_v = (weights[:, 1:, :] - weights[:, :-1, :]).norm(2)
        loss += alpha * (diff_h + diff_v)    
    return loss


def train(model, optimizer, loader, alpha, timewindow=10, model_type='LR', loss_type='bce', reg_type=None, collect_result=False, device='cuda'):
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
        loss = get_loss(model, labels, outputs, alpha=alpha, timewindow=timewindow, loss_type=loss_type, reg_type=reg_type, reduction='sum')
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
        return epoch_loss, acc

def evaluate(model, optimizer, loader, alpha, timewindow=10, model_type='LR', loss_type='bce', reg_type=None, collect_result=False, device='cuda'):
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
            loss = get_loss(model, labels, outputs, alpha=alpha, timewindow=timewindow, loss_type=loss_type, reg_type=reg_type, reduction='sum')
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
    

def plot_weight_glm(device, path, model_type, CH, loss_type, reg_type, alpha, best_epoch, timewindow=10):
    if CH == 'all':
        model = GLM().to(device)
        model.load_state_dict(torch.load('{}/{}_CH{}_LOSS{}_REG{}{}_EPOCH{}_REDUCEsum.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, best_epoch)))
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
        model.load_state_dict(torch.load('{}/{}_CH{}_LOSS{}_REG{}{}_EPOCH{}_REDUCEsum.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, best_epoch)))
        weights = model.linear.weight.view(100, timewindow)
        weights_sub = weights.detach().cpu()
        plt.yticks(ticks=[0, 20, 40, 60, 80, 99], labels=[round(np.logspace(0, 2.45, 100)[i]) for i in [0, 20, 40, 60, 80, 99]])
        plt.xticks(ticks=[0, timewindow-1], labels=[1, timewindow])
        plt.imshow(weights_sub)
        plt.title(str(CH))
        plt.show()
        
def plot_confusion(test_preds, test_labels):
    predictions_test = np.concatenate(test_preds)
    labels_test = np.concatenate(test_labels)

    cm_test = confusion_matrix(labels_test, predictions_test)
    cm_test_percent = (cm_test.T/cm_test.astype(np.float).sum(axis=1)).T
    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    sn.heatmap(cm_test, annot = True,  fmt = 'd', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.subplot(1,2,2)
    sn.heatmap(cm_test_percent, annot = True, cmap='Blues')
    plt.title('Test Confusion Matrix (Rate)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
        
        
def tuning(train_loader, val_loader, model, optimizer, device, num_epochs, alpha, model_type, loss_type, reg_type, CH, path, timewindow=10, verbose=False):
    training_losses, training_acc, val_losses, validation_acc = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, train_loader, alpha=alpha, timewindow=timewindow, model_type=model_type, loss_type=loss_type, reg_type=reg_type, collect_result=False, device=device)
        val_loss, val_acc = evaluate(model, optimizer, val_loader, alpha=alpha, timewindow=timewindow, model_type=model_type, loss_type=loss_type, reg_type=reg_type, collect_result=False, device=device)
        training_losses.append(train_loss)
        training_acc.append(train_acc)
        val_losses.append(val_loss)
        validation_acc.append(val_acc)
        if val_loss <= min(val_losses):
            best_epoch = epoch
            print(epoch)
            print('Train loss for epoch {}: {}'.format(epoch, train_loss))
            print('Val loss for epoch {}: {}'.format(epoch, val_loss))
            torch.save(model.state_dict(), '{}/{}_CH{}_LOSS{}_REG{}{}_EPOCH{}_REDUCEsum.pt'.format(path, model_type, CH, loss_type, reg_type, alpha, epoch))
        elif verbose:
            print('Train loss for epoch {}: {}'.format(epoch, train_loss))
            print('Val loss for epoch {}: {}'.format(epoch, val_loss))
    plot_loss_acc(training_losses, val_losses, training_acc, validation_acc, model_type)
    plot_weight_glm(device, path, model_type, CH, loss_type, reg_type, alpha, best_epoch)  
    return best_epoch, min(val_losses)
    plot_weight_glm(device, path, model_type, CH, loss_type, reg_type, alpha, best_epoch)
    