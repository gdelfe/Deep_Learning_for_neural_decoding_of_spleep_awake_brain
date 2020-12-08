"""
Full implementation of Krishan's labeling on our datset.


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


save_path = '/mnt/pesaranlab/People/Capstone_students/Noah/datav5/'


window = 10

nights = ['180326','180327','180328','180329','180330','180331',
		'180401','180409','180410','180411','180412','180413','180414'] # all that are processed. I think it's enough. 
recs = ['001','002','003','004','005','006','007','008','009']

for night in nights:
    for rec in recs:
        print("Checking ", night, ' ', rec)

        try: 
            # Add all other channels too, and concat. 
            NSV = loadmat('/vol/sas2b/Goose_Multiscale_M1_Wireless/'+night+'/'+rec+'/rec'+rec+'.NightStateVars.mat')
            movement_states = NSV['UserData'][0][0]['NeuralRecMovementStates'][0]
            sleep_states = NSV['UserData'][0][0]['NeuralRecSleepStates'][0]
            spec_data = loadmat('/mnt/pesaranlab/People/Capstone_students/Spectrogram_mat_data/N10W1dn1_'+night+'_rec'+rec+'.mat')
            specs = []
            all_badtimes = np.array([])

            for ch in range(1,63): #iterate over the 62 channels saved, stack together and proceed with that. 
                ztotSpec = spec_data['Spec_per_Ch']['Ch'+str(ch)][0][0]['ztotSpec'][0][0]
                specs.append(ztotSpec)
                badtimes = spec_data['Spec_per_Ch']['Ch'+str(ch)][0][0]['badtimes'][0][0]
                all_badtimes = np.concatenate([all_badtimes, badtimes.flatten()])
    
            
            # Compare these two, 
            ztotSpecs = np.stack(specs)
            all_badtimes = np.array(list(set(all_badtimes)))
            print('zts.shape=',ztotSpecs.shape)
            #NSVztotSpec = NSV['UserData'][0][0]['ztotSpec']
            movement_states_idx = []

            for state in movement_states:
                start = state[0][0]
                start_idx =  (np.abs(times-start)).argmin()
                stop = state[0][1]
                stop_idx = (np.abs(times-stop)).argmin()
                
                movement_states_idx.append([start_idx, stop_idx])
                
            sleep_states_idx = []

            for state in sleep_states:
                start = state[0][0]
                start_idx =  (np.abs(times-start)).argmin()
                stop = state[0][1]
                stop_idx = (np.abs(times-stop)).argmin()
                
                sleep_states_idx.append([start_idx, stop_idx])

            for badtime in all_badtimes:
                 idx = (np.abs(times-badtime)).argmin()
                 badtime_idx.append(idx)

            # for all bad times, imput NaN 
            for idx in badtime_idx:
                 ztotSpecs[:,idx,:] = np.nan



            #TODO: For camera fall, out of frames.

            # Saving in intervals.. 
            for s in sleep_states_idx:
                start = s[0]
                stop = s[1]


                sspec = torch.from_numpy(ztotSpecs[:,start:stop,:])
                sspecs = torch.split(sspec,window,dim=1)

                for i, arr in enumerate(sspecs):
                    if not torch.isnan(arr.sum()) and arr.shape[1] == window:
                        np.save(save_path + 'sleep/' + night + '_' + rec + '_' + str(ch) + '_win'+str(i) +'_sleep.npy',arr.numpy())
   
            for s in move_states_idx:
                start = s[0]
                stop = s[1]


                sspec = torch.from_numpy(ztotSpecs[:,start:stop,:])
                sspecs = torch.split(sspec,window,dim=1)

                for i, arr in enumerate(sspecs):
                    if not torch.isnan(arr.sum()) and arr.shape[1] == window:
                        np.save(save_path + 'move/' + night + '_' + rec + '_' + str(ch) + '_win'+str(i) +'_move.npy',arr.numpy())
   

        except Exception as e:
            print(e)
        
total_nonmvmt_samples = len(os.listdir(save_path + 'sleep')) 
total_mvmt_samples = len(os.listdir(save_path + 'move')) 
        
print("Done! Total non-movement = ", total_nonmvmt_samples, ' and total movement = ', total_mvmt_samples)



