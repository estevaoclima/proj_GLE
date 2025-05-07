#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:20:56 2025


testing if cloned repository is working fine


@author: estevao
"""

import os
from func_binary_to_mat_v06_05may25 import bin2mat
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
#import h5py
import pickle


# ------------

# 1) CHOOSE RECORDING SITE & SAVING PATH


recordingSite = 'gle05d'

# where raw data is stored
searchPath = '/home/estevao/Documents/visLab/proj_gle_cat_2025/gle/'

# where new data should be stored
outputPath = f'/home/estevao/Documents/visLab/analise_GLE/data4sorting/spkData_{recordingSite}/'


'''
# DEU ERRO NO ARQUIVO
arquivos com erro : inseri na pasta: badData_estevao30marco
'gle03a16'
'gle04c34'
'gle04c44'
'gle04c52'
'gle04f01'
'gle04f02'
'gle04f20'
'gle04f27'
'gle04i10'
'gle05b02'
'gle05b13'
'gle05b14'
'gle05d09'
'''


# ------------

# 2) FOR EACH RECORD FROM THIS RECORDING SITE, CREATE OBJECT OF EACH WITH ALL ITS INFOS
# SAVE AS .mat & .npz (python)

# create readableData in .mat & .npz
for record in os.listdir(searchPath):
    if recordingSite in record:
        print(f'\nGet data from Record: {record}')
        folderPath = searchPath+record
        os.chdir(folderPath)

        # GET DATA
        singleRecord = bin2mat(record, folderPath, recordingSite)

        # SAVE EACH RECORD
        os.makedirs(outputPath, exist_ok=True)  # create dir; false=prevents creating a copy/overwritting
        # save Pickle (good PYTHON format for mixed-types and nested data)
        with open(outputPath+f'{record}_data.pkl', 'wb') as f:
            pickle.dump(singleRecord, f)
        # save Python format
        #np.savez(outputPath+f'{record}_data.npz',singleRecord)        
        # save on MATLAB format
        #savemat(outputPath+f'{record}_data.mat', singleRecord)
        
        # adapt data to matlab (just 2 the nested data)
        def dict_to_cell_array(d):
            '''Necessary conversion from nested dict into cell for matlab compatibility.'''
            max_key = max(d.keys()) if d else 0
            cell_array = np.empty((max_key + 1, 1), dtype=object)
            
            for k, v in d.items():
                cell_array[k, 0] = v if v is not None else np.nan
            
            return cell_array
        # adapt data to matlab (just 2 the nested data)
        if singleRecord['num_of_trls'] != 1:
            matlab_index_perTrial = dict_to_cell_array(singleRecord['index_perTrial'])
            matlab_index_perTrial = matlab_index_perTrial[1:, 0]
            matlab_spk_perTrial = dict_to_cell_array(singleRecord['spk_perTrial'])
            matlab_spk_perTrial = matlab_spk_perTrial[1:, 0]
        else:
                matlab_index_perTrial = singleRecord['index_perTrial']
                matlab_spk_perTrial = singleRecord['spikes']
        
        savemat(outputPath+f'{record}_data.mat', {
            #general info
            'record': singleRecord['record'],
            'time_ms': singleRecord['time_ms'],
            'num_of_trls': singleRecord['num_of_trls'],
            'numSpk_perTrial': singleRecord['numSpk_perTrial'],

            # timeStamps (spike time)
            'index': singleRecord['index'],
            'index_continuous': singleRecord['index_continuous'],
            'index_perTrial': matlab_index_perTrial,  # singleRecord['index_perTrial'],

            # waveform
            'spk_perTrial': matlab_spk_perTrial,  #singleRecord['spk_perTrial'],
            'spikes': singleRecord['spikes'],

            # metadata
            'analog': singleRecord['analog'],
            'bhv': singleRecord['bhv'],
            'stimConditions': singleRecord['stimConditions']
            })



print('\n\nSaved all records as .mat & .pkl!!!!   :)')



# ------------

# 3) CONCATENATE ALL FILES FROM A SINGLE RECORDING SITE (to sort USING all)

# create grouped data for SpikeSorting from this single recording Site
all_index = []
all_index_cont = []
all_spikes = []

all_info = []
last_trl_cont = 0  # use when grouping continuous data

for record in os.listdir(outputPath):
    if recordingSite in record and '.pkl' in record:
        print(f'\nGet data from Record: {record}')
        with open(outputPath+f'{record}', 'rb') as f:
            data = pickle.load(f)
        # files_values = data.values()  # look at available data
        # files_values = data.__dict__
        # Get data
        single_index = data['index']
        single_index_cont = data['index_continuous']
        single_spikes = data['spikes']
        spks_perTrl = data['numSpk_perTrial']
        trl_info = []
        trl_info.append(spks_perTrl)
        trl_info.append(record[:8])


        # append it to the RecordingSite data
        all_index.append(single_index)
        all_spikes.append(single_spikes)
        all_info.append(trl_info)
        
        # add last value to continuesData to create gaps btw each
        if type(single_index_cont[0]) is not str:# np.str_:
            add_last_trl = single_index_cont + last_trl_cont
            all_index_cont.append(add_last_trl)
            last_trl_cont = all_index_cont[:][0][-1] + 1000
        else:  # it is a single trial
            add_last_trl = single_index + last_trl_cont  # it is a single trial
            all_index_cont.append(add_last_trl)
            last_trl_cont = all_index_cont[:][0][-1] + 1000





recSite_index = np.concatenate(all_index)
recSite_spikes = np.concatenate(all_spikes)
recSite_index_cont = np.concatenate(all_index_cont)



info_df = pd.DataFrame(all_info, columns=['num_spks_perTrial', 'record'])



print('\n\nConcat. from this single recording-site!')




# ------------

# 4) SAVE .mat FILE from this recording-site (all records from this)

# save on matlab format (as will be used in wave_clus)
savemat(
        outputPath+f'data4sorting_{recordingSite}.mat', {
            'index_individualTrials': recSite_index,
            'spikes': recSite_spikes,
            'numberOf_spks_perTrial':info_df,
            'index': recSite_index_cont
            })
print('\n\n-----------\n\nSaved all concat. as a single recording-site data for sorting!!!!)')




'''

searching for highest_point/peak


peak_all = []
for spk_idx in range(np.size(recSite_spikes,0)):
    np.argmax(recSite_spikes[0],1)
    peak_position = pd.Series(recSite_spikes[spk_idx]).idxmax()
    peak_all.append(peak_position)

asd = np.unique(np.array(peak_all))


pd.Series(recSite_spikes[0]).idxmax()

recSite_spikes
spk_idx = 0

np.argmax(recSite_spikes[0],1)

recSite_spikes[0]



zzzasd = np.argmax(recSite_spikes,1)

zxc, zzcxv, zzzzz = np.unique(zzzasd, return_counts=True, return_index=True)

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(15, 5))
ax = sns.lineplot(recSite_spikes[321],
                  linewidth=6.6,
                  alpha=1,
                  color='black')
'''

