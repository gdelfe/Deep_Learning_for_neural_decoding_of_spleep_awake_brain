"""

Create dataset of 10 second windows for neural state classification as either movement or non-movement (sleep)


Depends on loading in Krishan's NightStateVars.mat files which contain the periods of movement and non-movement




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

save_path = '/mnt/pesaranlab/People/Capstone_students/Noah/datav3/'
ch = 'na'
window = 10 # 10 sec interval 
# Loop through dates, and recs
nights = ['180326','180327','180328','180329','180330','180331'] # all that are processed. I think it's enough. 
recs = ['001','002','003','004','005','006','007','008','009']
for night in nights:
    for rec in recs:
        print("Checking ", night, ' ', rec)
        try: 
            NSV = loadmat('/vol/sas2b/Goose_Multiscale_M1_Wireless/'+night+'/'+rec+'/rec'+rec+'.NightStateVars.mat')



            ztotSpec = NSV['UserData'][0][0]['ztotSpec']

            sleep = NSV['UserData'][0][0]['SleepStates_noart'][0][:]
            move = NSV['UserData'][0][0]['MovementStates_noart'][0][:]


            for m in move:
                start = m[0][0] 
                stop = m[0][1]

                mspec = torch.from_numpy(ztotSpec[:,start:stop])
                mspecs = torch.split(mspec,10,dim=1)

                # save each

                for i, arr in enumerate(mspecs):
                    if arr.shape[1] == window:
                        np.save(save_path + 'move/' + night + '_' + rec + '_' + str(ch) + '_win'+str(i) +'_move.npy',arr.numpy())



            for s in sleep:
                start = s[0][0] 
                stop = s[0][1]

                sspec = torch.from_numpy(ztotSpec[:,start:stop])
                sspecs = torch.split(sspec,10,dim=1)

                # save each

                for i, arr in enumerate(sspecs):
                    if arr.shape[1] == window:
                        np.save(save_path + 'sleep/' + night + '_' + rec + '_' + str(ch) + '_win'+str(i) +'_sleep.npy',arr.numpy())

        except Exception as e:
            print(e)
            continue
