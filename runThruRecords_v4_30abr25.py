#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:20:56 2025

@author: estevao
"""

import os
from func_binary_to_mat_v05_30abr25 import bin2mat
import numpy as np
from scipy.io import savemat






# ------------

# 1) CHOOSE RECORDING SITE



recordingSite = 'gle05c'


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

# 2) FIND INDIVIDUAL RECORDINGS AND SAVE AS .mat & .npz (python)


searchPath = '/home/estevao/Documents/visLab/proj_gle_cat_2025/gle/'


# create readableData in .mat & .npz
for record in os.listdir(searchPath):
    if recordingSite in record:
        print(f'\nGet data from Record: {record}')
        folderPath = searchPath+record
        os.chdir(folderPath)

        # GET DATA
        bin2mat(record, folderPath, recordingSite)


print('\n\nGravou todos .mat & .npz ate o final!!!!   :)')








# ------------

# 3) CONCATENATE ALL FILES FROM A SINGLE RECORDING SITE (to sort USING all)


searchPath = f'/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting/spkData_{recordingSite}/'
#recordingSite = 'gle04b'


# create grouped data for SpikeSorting from this single recording Site
all_index = []
all_index_cont = []
all_spikes = []

all_info = []
last_trl_cont = 0  # use when grouping continuous data

for record in os.listdir(searchPath):
    if recordingSite in record and '.npz' in record:
        print(f'\nGet data from Record: {record}')

        # load npz data
        data = np.load(searchPath+record)

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
        if type(single_index_cont[0]) is not np.str_:
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


import pandas as pd
info_df = pd.DataFrame(all_info, columns=['num_spks_perTrial', 'record'])



print('\n\nCONCATENOU todos!')


# SAVE ON .mat FILE
outputPath_mat = f'/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting/spkData_{recordingSite}/'
os.makedirs(outputPath_mat, exist_ok=True)  # create dir; false=prevents creating a copy/overwritting

# save on matlab format
savemat(
        outputPath_mat+f'data4sorting_{recordingSite}.mat', {
            'index_individualTrials': recSite_index,
            'spikes': recSite_spikes,
            'numberOf_spks_perTrial':info_df,
            'index': recSite_index_cont
            })
print('\n\n-----------\n\nSalvou concatenados para sorting!!!!)')




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

