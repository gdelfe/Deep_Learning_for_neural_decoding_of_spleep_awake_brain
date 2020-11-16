"""
.py file implementation of data collection technique, much easier to deal with. 

"""


# import dependencies

import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import torch
import os
from scipy.stats import zscore
from scipy.io import loadmat



# first, a util function used if there are no transitions in the currently loaded in NE file


def no_transitions(file,night,rec,spec_path):
    ## Given a file, figure out what label it is for entire recording, save it all as that
    try: 
        prev_file = 'N10W1dn1_'+night+'_rec'+ '00' + str(int(rec) - 1)+'.mat'
        
        # load in prev file, check transitions
        if os.path.exists(spec_path + prev_file):
            prev_spec_data = loadmat(spec_path + prev_file)
            ti = prev_spec_data['Spec_per_Ch']['Ch1'][0][0]['ti'][0][0]
            vid_stop = ti[0][-1]/ 1000
            m_stop = prev_spec_data['Spec_per_Ch']['Ch1'][0][0]['m_stop'][0][0][0][-1] #[0][0][6]
        
            if m_stop < vid_stop:
                print('Monkey is sleeping for the entire new recording!')
                
                
                return np.array([0]), np.array([0])

    except:
        print("prev rec did not work. how about two recordings prior? ")
        prev_file = 'N10W1dn1_'+night+'_rec'+ '00' + str(int(rec) - 2)+'.mat'
        
        # load in prev file, check transitions
        if os.path.exists(spec_path + prev_file):
            prev_spec_data = loadmat(spec_path + prev_file)
            ti = prev_spec_data['Spec_per_Ch']['Ch1'][0][0]['ti'][0][0]
            vid_stop = ti[0][-1]/ 1000
            m_stop = prev_spec_data['Spec_per_Ch']['Ch1'][0][0]['m_stop'][0][0][0][-1] #[0][0][6]
        
            if m_stop < vid_stop:
                print('Monkey is sleeping for the entire new recording!')
                
                
                return np.array([0]), np.array([0])

def create_spec_dataset(spec_path = '../../../Spectrogram_mat_data/', save_path = '../../data/', task='movement',window = 10):
    """

    Loads in, the night event files, and uses those to then split into whatever designated window size is 
    used to then save in separate folders the corresponding input and output pairs for the movement vs non-movement classification task. 


    Would definitely be worth including as an attribute what the downstream task is, so as
    to allow this method to be used for saving any form of the data. 




    """



    #TODO: load in specs


    for file in os.listdir(spec_path)[:]:
        if file[-4:] != '.mat': # only consider .mat files 
            continue
        # sort of a metadata collection, read in what night and rec # this file corresponds to. 
        night = file.split('_')[1]
        rec = file.split('_')[2].split('.')[0][-3:]
        print("currently reading in ",file)

        ch = 0
        # TODO: Read spec main data
        spec_data = loadmat(spec_path + file)


        try:
            ti = spec_data['Spec_per_Ch']['Ch1'][0][0]['ti'][0][0]
        except Exception as e: 
            print("Error when trying to load in ti file. Something's wrong with ", file)
            print(e)
            continue

        # load in the other attributes. 
        f = spec_data['Spec_per_Ch']['Ch1'][0][0]['f'][0][0]#[0][0][ch][0][0][1]
        times = ti[0][:] // 1000
        badtimes = spec_data['Spec_per_Ch']['Ch1'][0][0]['badtimes'][0][0]#[0][0][ch][0][0][3]
        m_start = spec_data['Spec_per_Ch']['Ch1'][0][0]['m_start'][0][0]
        m_stop = spec_data['Spec_per_Ch']['Ch1'][0][0]['m_stop'][0][0]#[0][0][6]
        if m_start.shape[0] == 0: #check to see if there is movement info as part of this rec. 


           # print("Skipping ", file, " (no mstart)")
            print("This spec is either entirely moving or not moving. So let's check the prior rec. ")
            try:
                m_start, m_stop = no_transitions(file,night,rec,spec_path)  #TODO: add night before too, in case both are no transition. 
                print("Was able to use prior rec, saving.")
            except Exception as e:
                print(e)
                print("Unable to read ", file)
                continue
        else:
            m_start = m_start[0]
            m_stop = m_stop[0]



        specs = []
        for ch in range(1,63): #iterate over the 62 channels saved, stack together and proceed with that. 
            ztotSpec = spec_data['Spec_per_Ch']['Ch'+str(ch)][0][0]['ztotSpec'][0][0]
            specs.append(ztotSpec)
        ztotSpecs = np.stack(specs)

        # transforms the movement start and stop times into the corresponding index on the 
        # spectrogram. 

        m_start_idx = []  
        for start in m_start:
            idx = (np.abs(times-start)).argmin()
            m_start_idx.append(idx)

        m_stop_idx = []
        for stop in m_stop:
            idx = (np.abs(times-stop)).argmin()
            m_stop_idx.append(idx)


        # transforms the badtime time into corresponding index on spectrogram. In doing so, converts
        # that entire col to NaN values to ensure that window is discarded during saving. 
        badtime_idx = []

        for badtime in badtimes[0]:
            idx = (np.abs(times-badtime)).argmin()
            badtime_idx.append(idx)

        for idx in badtime_idx:
            ztotSpecs[:,idx,:] = np.nan



        prior_mvmt_idx = 0 #initialize to 0, saves everything preceeding mvmt start idx to be non-movement
        for start_idx, stop_idx in zip(m_start_idx,m_stop_idx):

            asleep = torch.from_numpy(ztotSpecs[:,int(round(prior_mvmt_idx)):int(round(start_idx)),:])
            moving = torch.from_numpy(ztotSpecs[:,int(round(start_idx)):int(round(stop_idx)),:])
            prior_mvmt_idx = stop_idx


            asleeps = torch.split(asleep,split_size_or_sections=window,dim=1)
            movings = torch.split(moving,split_size_or_sections=window,dim=1)

            print("Total # of non-movement windows saved: ", len(asleeps)-1) #not exact, but still a good indicator for this phase. 
            print("Total # of movement windows saved: ", len(movings)-1)

            # Now of these two arrays, split and save, assuming it does not include a badtime index

                                # loop over these windows and save 
            for i, arr in enumerate(asleeps):
                if not torch.isnan(arr.sum()) and arr.shape[1] == window:
                    np.save(save_path + 'sleep/' + night + '_' + rec + '_' + str(ch) + '_win'+str(i) +'_sleep.npy',arr.numpy()) 
            for i, arr in enumerate(movings):
                if not torch.isnan(arr.sum()) and arr.shape[1] == window:
                    np.save(save_path + 'move/' + night + '_' + rec + '_' + str(ch) + '_win'+str(i) +'_move.npy',arr.numpy()) 

    total_nonmvmt_samples = len(os.lisdir(save_path + 'sleep')) 
    total_mvmt_samples = len(os.lisdir(save_path + 'move')) 
           	
    print("Done! Total non-movement = ", total_nonmvmt_samples, ' and total movement = ', total_mvmt_samples)


if __name__ == '__main__':
    create_spec_dataset()