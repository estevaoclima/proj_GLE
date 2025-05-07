#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:41:44 2025

@author: estevao





ORGANIZE DATA POST SORTING infos






"""



import struct
import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

# -----------------

# 1 - Choose Record

record = 'gle04e06'


conditions = 5

pre_stim_duration = 1000#2000
stim_duration = 2000
post_stim_duration = 2000#2000
trial_duration = post_stim_duration+stim_duration+pre_stim_duration



# -----------------

# 2 - LOAD DATA

recordingSite = record[:-2]

# POS ANALISE NO LAB (vpnUSP hackeado)
folderPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}/'
sortData = loadmat(folderPath+f'times_novissimo_data4sorting_{recordingSite}.mat')
preSortData = loadmat(folderPath+f'data4sorting_{recordingSite}.mat')



loadmat(folderPath+f'data4sorting_{recordingSite}.mat')



wfm = sortData['spikes']
spk_clusId_Time = sortData['cluster_class']
numspk_perTrl = preSortData['numberOf_spks_perTrial']#[4][0][0]
spk_idxIndiv = preSortData['index_individualTrials'].T



# -----------
# .stim

searchPath = '/home/estevao/Documents/visLab/proj_gle_cat_2025/gle/'
folderPath = searchPath+record
os.chdir(folderPath)
fileSpk = folderPath+'/'+record+'.stim'
filename = fileSpk
"""
Get .stim file knowning it has only int32 values (info from spass2field fieldtrip).
"""
with open(filename, 'rb') as f:
    spkData = f.read()  # read the entire file
    fileSize_bytes = len(spkData)  # size in BYTES
    fileSize_values = fileSize_bytes // 4#4  # as I know it is int32 (1 values==4 bytes )
    metadata_stim = struct.unpack(f'>{fileSize_values}i', spkData)

thisRec_condinfo = np.array(metadata_stim)[1:]






## CREATE IDENTIFIER FOR PRE-STIM | STIM | POST-STIM (0,1,2)

thisRec_trlPeriod = []  # it seems like there is an ERROR in time control...

for jj in range(np.size(spk_idxIndiv, 0)):
    if spk_idxIndiv[jj] < pre_stim_duration:
        moment = 0  # pre-stim period
    elif spk_idxIndiv[jj] >= pre_stim_duration:
        if spk_idxIndiv[jj] < (pre_stim_duration + stim_duration):
            moment = 1  # stim period
        else:
            moment = 2  # post-stim period
    
    thisRec_trlPeriod.append(moment)





# 3 - CREATE DATA FRAME - (organize data set)


# isolate each clus per trial
#spks_perRecord = np.sum(np.array(numspk_perTrl)[0,0])
records_numSpks = [
    np.sum(np.array(numspk_perTrl)[jj, 0])
    for jj in range(np.size(numspk_perTrl, 0))
    ]

records_names = [
    str(np.array(numspk_perTrl)[jj, 1])[2:-2]
    for jj in range(np.size(numspk_perTrl, 0))
    ]

# split sorting_id & wfm data info per record
cluster_idx_perRecord = []
spkTime_perRecord = []
wfm_perRecord = []

lastCount = 0
for jj in range(np.size(records_numSpks, 0)):
    #aa = spk_clusId_Time[lastCount:(records_numSpks[jj]+lastCount)]
    aa = np.array(spk_clusId_Time)[lastCount:(records_numSpks[jj]+lastCount), 0]
    bb = np.array(spk_clusId_Time)[lastCount:(records_numSpks[jj]+lastCount), 1]
    cc = wfm[lastCount:(records_numSpks[jj]+lastCount)]

    lastCount = records_numSpks[jj]  # keep track of where you are

    cluster_idx_perRecord.append(aa)
    spkTime_perRecord.append(bb)
    wfm_perRecord.append(cc)



data_df = pd.DataFrame({
    'record': records_names,
    'tot_spks': records_numSpks,
    'numspk_perTrl': np.array(numspk_perTrl)[:,0],
    #'record':str(np.array(numspk_perTrl)[:,1])[2:-2],
    'wfm': wfm_perRecord,
    'cluster_id': cluster_idx_perRecord,
    'spk_time': spkTime_perRecord
    })













