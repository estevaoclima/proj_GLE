#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:53:09 2025

@author: estevao
"""


from scipy.io import loadmat
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

# -----------------

# 1 - Choose Record
record = 'gle05c03'#'gle04d02'
record = 'gle05a05'#'gle04d02'

record = 'gle05a03'#'gle05c03'#'gle04d02'
record = 'gle04i11'

record = 'gle04e06'


conditions = 5

pre_stim_duration = 1000#2000
stim_duration = 2000
post_stim_duration = 2000#2000
trial_duration = post_stim_duration+stim_duration+pre_stim_duration

# -----------------

# 2 - LOAD DATA

recordingSite = record[:-2]
#folderPath = '/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting/spkData_gle04d/'
#folderPath = f'/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting/spkData_{recordingSite}/'
#sortData = loadmat(folderPath+f'times_novissimo_data4sorting_{recordingSite}.mat')
#preSortData = loadmat(folderPath+f'data4sorting_{recordingSite}.mat')

#sortData = loadmat(folderPath+'times_novissimo_data4sorting_gle04d.mat')
#preSortData = loadmat(folderPath+'data4sorting_gle04d.mat')
#single_recordData = loadmat(folderPath+'gle04d02_data4matlab.mat')

# POS ANALISE NO LAB (vpnUSP hackeado)
folderPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}/'
sortData = loadmat(folderPath+f'times_novissimo_data4sorting_{recordingSite}.mat')
preSortData = loadmat(folderPath+f'data4sorting_{recordingSite}.mat')



loadmat(folderPath+f'data4sorting_{recordingSite}.mat')



wfm = sortData['spikes']
spk_clusId_Time = sortData['cluster_class']
numspk_perTrl = preSortData['numberOf_spks_perTrial']#[4][0][0]
spk_idxIndiv = preSortData['index_individualTrials'].T





## CREATE IDENTIFIER FOR PRE-STIM | STIM | POST-STIM (0,1,2)

thisRec_trlPeriod = []  # it seems like there is an ERROR in time control...

for jj in range(np.size(spk_idxIndiv, 0)):
    if spk_idxIndiv[jj] < pre_stim_duration:
        moment = 0
    elif spk_idxIndiv[jj] >= pre_stim_duration:
        if spk_idxIndiv[jj] < (pre_stim_duration + stim_duration):
            moment = 1
        else:
            moment = 2
    
    thisRec_trlPeriod.append(moment)






# -----------------

# 2b - CREATE FUNCTION FO RAUTOCORRELATION

def autocorrelation_analysis(spike_times, bin_size=1, window=100, normalize=True):
    """
    Compute autocorrelation of spike times using scipy.
    
    Parameters:
    -----------
    spike_times : array-like
        Array of spike times in ms
    bin_size : float
        Size of bins for autocorrelation (ms)
    window : float
        Window size for autocorrelation (ms)
    normalize : bool
        Whether to normalize the autocorrelation
    
    Returns:
    --------
    lags : array
        Lag times (ms)
    acorr : array
        Autocorrelation values
    """
    if len(spike_times) == 0:
        print("Warning: Empty spike array provided for autocorrelation")
        return np.array([]), np.array([])
    
    # Convert spike times to a binary array
    max_time = np.ceil(max(spike_times))
    min_time = np.floor(min(spike_times))
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    
    # Handle edge case where all spikes are at the same time
    if len(bins) <= 1:
        print("Warning: All spikes occur at same time, autocorrelation not meaningful")
        return np.array([0]), np.array([1])
    
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    
    # Compute autocorrelation using scipy.signal.correlate
    acorr = signal.correlate(spike_counts, spike_counts, mode='full')
    
    # Normalize if requested
    if normalize:
        if np.max(acorr) > 0:  # Avoid division by zero
            acorr = acorr / np.max(acorr)
    
    # Compute lag times
    lags = np.arange(-len(spike_counts) + 1, len(spike_counts)) * bin_size
    
    # Limit to specified window
    window_idx = np.where(np.abs(lags) <= window)[0]
    if len(window_idx) > 0:
        lags = lags[window_idx]
        acorr = acorr[window_idx]
    
    return lags, acorr




# -----------------

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




# -----------------

# 4 - SELECT DATA FROM CHOOSEN RECORD

selectedData = data_df[data_df['record'] == record]





# -----------------

# 5 - DISTRIBUTE DATA AMONG CONDITIONS
# create a single record dataframe


# GET ONLY ONE CONDITION
# reshape the number_of_spks_perTrl
thisRec_numspk_perTrl = (selectedData['numspk_perTrl'].values[0])
thisRec_idx_clus = selectedData['cluster_id'].values[0]
thisRec_spkTime = selectedData['spk_time'].values[0]
thisRec_wfm = selectedData['wfm'].values[0]




condRepetitions = int(np.size(thisRec_numspk_perTrl)/conditions)
thisRec_condinfo = np.tile((np.arange(conditions)+1), condRepetitions)



## obs, geting the true stim conditions order 
# got this from another code that I run before (will implement this later on this code)
thisRec_condinfo = np.array(metadata_stim)[1:]

















# organize this record data per trials (to know which conditions we'll get)
thisRec_idx_clus_reshape = []
thisRec_spkTime_reshape = []
thisRec_wfm_reshape = []
thisRec_trlPeriod_reshape = []

allTrlsBefore = 0
for column_idx in range(np.size(thisRec_numspk_perTrl,1)):
    #ii = np.array(thisRec_numspk_perTrl)[0,column_idx]
    ii = np.array(thisRec_numspk_perTrl)[0,column_idx]
    qq_clusId = np.array(thisRec_idx_clus)[allTrlsBefore:ii+allTrlsBefore]
    qq_spkTime = np.array(thisRec_spkTime)[allTrlsBefore:ii+allTrlsBefore]
    qq_wfm = np.array(thisRec_wfm)[allTrlsBefore:ii+allTrlsBefore]
    qq_trlPer = np.array(thisRec_trlPeriod)[allTrlsBefore:ii+allTrlsBefore]
    
    
    #allTrlsBefore = sum(np.array(thisRec_numspk_perTrl)[:,column_idx])
    allTrlsBefore = np.sum(np.array(thisRec_numspk_perTrl)[0,0:column_idx+1])
    thisRec_idx_clus_reshape.append(qq_clusId)
    thisRec_spkTime_reshape.append(qq_spkTime)
    thisRec_wfm_reshape.append(qq_wfm)
    thisRec_trlPeriod_reshape.append(qq_trlPer)


#thisRec_numspk_perTrl_transp = thisRec_numspk_perTrl.T
thisRec_df = pd.DataFrame({
    'condition':thisRec_condinfo,
    'idx_clus': thisRec_idx_clus_reshape,
    'spkTime':thisRec_spkTime_reshape,
    'numspk_perTrl': list(thisRec_numspk_perTrl.T),
    'wfm': thisRec_wfm_reshape,
    'trlPeriod': thisRec_trlPeriod_reshape
    })





# -----------------

# 6 -  AUTOCORRELATION

# -------- SELECT JUST ONE CONDITION DATA


# -----------------

# 6a LOOOP THRU COND AND SUA


# RUN THRU CONDITIONS
uniqueCond = thisRec_df['condition'].unique()

all_lags = []
all_acorr = []
for cond_idx in range(len(uniqueCond)):
    
    cond_lags = []
    cond_acorr = []
    
    thisCond_df = thisRec_df[thisRec_df['condition']==uniqueCond[cond_idx]]

    thisCond_spkTime = np.concatenate(thisCond_df['spkTime'].values)#thisCond_df['spkTime'].values[0]
    thisCond_idx_clus = np.concatenate(thisCond_df['idx_clus'].values)#thisCond_df['idx_clus'].values[0]
    thisCond_numspk_perTrl = np.array(thisCond_df['numspk_perTrl'])#.values[0]

    thisCond_trlPeriod = np.concatenate(thisCond_df['trlPeriod'].values)#.values[0]


    # check the data type
    #print(type(thisCond_df['trlPeriod'].iloc[0]))

    # RUN THRU SUAs
    uniqueSua = np.unique(thisCond_idx_clus)

    for clus_idx in range(len(uniqueSua)):
        single_sua_spkTime = thisCond_spkTime[thisCond_idx_clus==uniqueSua[clus_idx]]
        single_sua_trlPeriod = thisCond_trlPeriod[thisCond_idx_clus==uniqueSua[clus_idx]]
        
        
        
        # SELECT 4 JUST DURING STIM PERIOD
        single_sua = single_sua_spkTime[single_sua_trlPeriod == 1]


# -----------------

        # 6b - AUTOCORRELATION FOR THIS DATA
        single_lags, single_acorr = autocorrelation_analysis(single_sua, bin_size=1, window=100, normalize=True)


        cond_lags.append(single_lags) # save each group of SUAs for this condition
        cond_acorr.append(single_acorr)

    all_lags.append(cond_lags)  #save all groups of SUAs for all possible conditions
    all_acorr.append(cond_acorr)



# dataframe: ROWS: conditions; COLUMN: SUAs
acorr_df = pd.DataFrame(all_acorr)  # acorr_df[SUAs][condition]
lags_df = pd.DataFrame(all_lags)  # prob wont use that many... (could keep just the 1st?)





# -----------------

# 6d FIGURE - PLOT IN LOOP (EACH FIG FOR EACH SUA/ SUBPLOTS FOR EACH CONDITION)
#fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for clus_idx in range(len(uniqueSua)):
    fig, axs = plt.subplots(1, len(uniqueCond), figsize=(100, 10), sharex=True)
#    fig, axs = plt.subplots(figsize=(40, 10))  # for each condition

    for cond_idx in range(len(uniqueCond)):

        # get info form this condition|SUA
        acorr = acorr_df[clus_idx][cond_idx]
        lags = lags_df[clus_idx][cond_idx]

        # subplots for each condition
        if len(lags) > 0:
            axs[cond_idx].plot(lags, acorr)
            axs[cond_idx].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            axs[cond_idx].set_ylabel('Autocorrelation')
            axs[cond_idx].set_title(f'Stimulus Autocorrelation (n={len(single_sua)} spikes)')
            axs[cond_idx].grid(True, alpha=0.3)
            axs[cond_idx].set_ylim(-0.01, 0.15)







