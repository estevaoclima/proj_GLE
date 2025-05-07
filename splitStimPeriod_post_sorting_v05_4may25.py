#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:53:09 2025

@author: estevao
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
# testing different sorting clusterings
folderPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}_otherSort/'
#folderPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}/'
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






# -----------------

# 2b - CREATE FUNCTION FO RAUTOCORRELATION



def autocorrelation_analysis(spike_times, bin_size=1, window=250, normalize=True):
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
#    
#    # Normalize if requested
#    if normalize:
#        if np.max(acorr) > 0:  # Avoid division by zero
#            acorr = acorr / np.max(acorr)
#    
#    # Compute lag times
#    lags = np.arange(-len(spike_counts) + 1, len(spike_counts)) * bin_size
#    
#    # Limit to specified window
#    window_idx = np.where(np.abs(lags) <= window)[0]
#    if len(window_idx) > 0:
#        lags = lags[window_idx]
#        acorr = acorr[window_idx]
    
    
    
    ## Adjusting to CLaude proposal:
    # Calculate the lags in ms
    lags = np.arange(-len(spike_counts) + 1, len(spike_counts)) * bin_size
    
    # Only return the central portion of the autocorrelation (within the window)
    central_idx = len(lags) // 2
    window_samples = int(window / bin_size)
    start_idx = central_idx - window_samples
    end_idx = central_idx + window_samples + 1
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(lags), end_idx)
    
    # Normalize if requested
    if normalize:
        # Standard normalization - divide by the value at zero lag
        if acorr[central_idx] > 0:  # Avoid division by zero
            acorr = acorr / acorr[central_idx]
    
    return lags[start_idx:end_idx], acorr[start_idx:end_idx]
    
  #  return lags, acorr




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













# ----------------- --------------------------------------------------------------








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
    

#    # Try running per each trial and geting mean
#    thisCond_df = thisRec_df[thisRec_df['condition']==uniqueCond[cond_idx]].reset_index()
#    for trl_idx, trl_df in thisCond_df.iterrows():
#        print(trl_df['index'])
#        trl_idx_clus = trl_df['idx_clus']
#        trl_spkTime = trl_df['spkTime']
#        trl_trlPeriod = trl_df['trlPeriod']
#        
#        # just during stim Period
#        trl_spkTime_stimPer = trl_spkTime[trl_trlPeriod==1]
#        trl_idx_clus_stimPer = trl_idx_clus[trl_trlPeriod==1]
#        
#        # RUN THRU SUAs
#        uniqueSua = np.unique(trl_idx_clus_stimPer)
#        
#        for clus_idx in range(len(uniqueSua)):
#        trl_spkTime_singleSua = trl_spkTime_stimPer[trl_idx_clus_stimPer==clus_idx]
#        # trying to adjust
#        trl_spkTime_singleSua_dif =  trl_spkTime_singleSua - np.min(trl_spkTime_singleSua)
#        #trl_spkTime_singleSua_dif = trl_spkTime_singleSua_dif[:49]
#        
#        # AUTOCORRELATION
#        single_lags, single_acorr = autocorrelation_analysis(trl_spkTime_singleSua_dif, bin_size=1, window=250, normalize=True)


#fig, axs = plt.subplots(1, 2, figsize=(15, 4), sharex=True)
#axs[0].plot(single_lags, single_acorr, linewidth = 0.5)
#for lag in range(0, int(max(lags)), 25):
#        if lag > 0:  # Skip the zero lag
#            axs.axvline(x=lag, color='r', linestyle='--', alpha=0.5)
#            axs.axvline(x=-lag, color='r', linestyle='--', alpha=0.5)

# LATER REMOVE ALL THIS PARTS BELLOW HERE....
    
    
    
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
        #single_lags, single_acorr = autocorrelation_analysis(single_sua, bin_size=1, window=100, normalize=True)
        # No normalization >
       # single_lags, single_acorr = autocorrelation_analysis(single_sua, bin_size=1, window=100, normalize=False)
        single_lags, single_acorr = autocorrelation_analysis(single_sua, bin_size=2, window=250, normalize=True)

        cond_lags.append(single_lags) # save each group of SUAs for this condition
        cond_acorr.append(single_acorr)

    all_lags.append(cond_lags)  #save all groups of SUAs for all possible conditions
    all_acorr.append(cond_acorr)



# dataframe: ROWS: conditions; COLUMN: SUAs
acorr_df = pd.DataFrame(all_acorr)  # acorr_df[SUAs][condition]
lags_df = pd.DataFrame(all_lags)  # prob wont use that many... (could keep just the 1st?)





# -----------------

# 6d FIGURE - PLOT IN LOOP (EACH FIG FOR EACH SUA/ SUBPLOTS FOR EACH CONDITION)

for clus_idx in range(len(uniqueSua)):
    fig, axs = plt.subplots(1, len(uniqueCond), figsize=(15, 4), sharex=True)

    # figure title
    fig.suptitle(f'SUA {clus_idx}')#, y=1.05, fontsize=12)

    for cond_idx in range(len(uniqueCond)):
        # get info form this condition|SUA
        acorr = acorr_df[clus_idx][cond_idx]
        lags = lags_df[clus_idx][cond_idx]

        # subplots for each condition
        if len(lags) > 0:
            axs[cond_idx].plot(lags, acorr, linewidth = 0.5)
            axs[cond_idx].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            if cond_idx == 0:
                axs[cond_idx].set_ylabel('Autocorrelation')
            axs[cond_idx].set_title(f'Condition {cond_idx+1}', pad=10)
            axs[cond_idx].grid(True, alpha=0.3)
            axs[cond_idx].set_ylim(-0.01, 0.20)
            axs[cond_idx].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()






'''



for clus_idx in range(len(uniqueSua)):
    fig, axs = plt.subplots(1, len(uniqueCond), figsize=(15, 4), sharex=True, sharey=True)
    
    # Get current SUA number
    sua_num = uniqueSua[clus_idx]
    
    # Calculate total spikes for this SUA across all conditions
    total_spikes = 0
    for cond_idx in range(len(uniqueCond)):
        thisCond_df = thisRec_df[thisRec_df['condition']==uniqueCond[cond_idx]]
        thisCond_idx_clus = np.concatenate(thisCond_df['idx_clus'].values)
        total_spikes += np.sum(thisCond_idx_clus == sua_num)
    
    fig.suptitle(f'SUA {sua_num} | Total spikes: {total_spikes} | Bin: 1ms', y=1.05)

    for cond_idx in range(len(uniqueCond)):
        ax = axs[cond_idx] if len(uniqueCond) > 1 else axs
        
        # Get autocorrelation data
        acorr = acorr_df[clus_idx][cond_idx]
        lags = lags_df[clus_idx][cond_idx]

        if len(lags) > 0:
            # Plot with scientific markings
            ax.plot(lags, acorr, linewidth=1.2, color='#1f77b4')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
            
            # Mark refractory period (2ms window)
            ax.axvspan(-2, 2, color='red', alpha=0.05)
            
            # Calculate and show condition-specific spike count
            thisCond_df = thisRec_df[thisRec_df['condition']==uniqueCond[cond_idx]]
            cond_spikes = np.concatenate(thisCond_df['spkTime'].values)
            cond_clus = np.concatenate(thisCond_df['idx_clus'].values)
            spike_count = np.sum(cond_clus == sua_num)
            
            # Add firing rate info (spikes/sec)
            cond_duration = len(thisCond_df) * trial_duration / 1000  # in seconds
            firing_rate = spike_count / cond_duration if cond_duration > 0 else 0
            
            ax.set_title(f'Cond {cond_idx+1}\n(n={spike_count})', pad=10)
            ax.text(0.95, 0.85, f'{firing_rate:.1f} Hz', 
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
            
            # Axis labels only on first/last subplots
            if cond_idx == 0:
                ax.set_ylabel('Norm. Correlation')
            ax.set_xlabel('Lag (ms)') if cond_idx == len(uniqueCond)-1 else None
            
            # Style adjustments
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.01, 0.15)

    plt.tight_layout()
    plt.show()












for clus_idx in range(len(uniqueSua)):
    fig, axs = plt.subplots(1, len(uniqueCond), figsize=(18, 4), sharex=True)  # Wider figure
    
    sua_num = uniqueSua[clus_idx]
    fig.suptitle(f'SUA {sua_num} - Autocorrelation with Dominant Frequencies', y=1.05)
    
    for cond_idx in range(len(uniqueCond)):
        ax = axs[cond_idx]
        acorr = acorr_df[clus_idx][cond_idx]
        lags = lags_df[clus_idx][cond_idx]
        
        if len(lags) > 1:  # Need at least 2 points for frequency analysis
            # 1. Plot original autocorrelation
            ax.plot(lags, acorr, linewidth=1.5, color='#1f77b4', label='Autocorr')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
            
            # 2. Calculate dominant frequencies
            sampling_interval = lags[1] - lags[0]  # in ms
            sampling_rate = 1000 / sampling_interval  # in Hz
            
            # Compute FFT of autocorrelation
            fft_values = np.fft.rfft(acorr)
            fft_freq = np.fft.rfftfreq(len(acorr), d=sampling_interval/1000)  # Convert to seconds
            
            # Find peaks in the power spectrum
            power_spectrum = np.abs(fft_values)**2
            peaks, _ = signal.find_peaks(power_spectrum, height=np.mean(power_spectrum)*1.5)
            
            # 3. Annotate dominant frequencies on plot
            for peak in peaks[:3]:  # Show top 3 frequencies
                freq = fft_freq[peak]
                period = 1000/freq if freq > 0 else 0  # in ms
                
                if 5 < freq < 200:  # Only show biologically plausible frequencies
                    ax.axvline(period, color='orange', linestyle=':', alpha=0.7, linewidth=1)
                    ax.axvline(-period, color='orange', linestyle=':', alpha=0.7, linewidth=1)
                    ax.text(period, ax.get_ylim()[1]*0.9, 
                            f'{freq:.1f}Hz', 
                            ha='center', va='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7))
            
            # 4. Add inset with power spectrum
            inset_ax = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            inset_ax.plot(fft_freq, power_spectrum, color='purple', linewidth=1)
            inset_ax.set_xlabel('Freq (Hz)')
            inset_ax.set_ylabel('Power')
            inset_ax.set_xlim(0, 100)  # Focus on 0-100 Hz range
            inset_ax.grid(True, alpha=0.3)
            
            # Mark identified peaks in inset
            for peak in peaks[:3]:
                if 5 < fft_freq[peak] < 100:
                    inset_ax.axvline(fft_freq[peak], color='red', linestyle=':', alpha=0.5)
            
            # Style main plot
            ax.set_title(f'Condition {cond_idx+1}\n(n={len(acorr)} points)', pad=10)
            ax.set_xlabel('Lag (ms)')
            if cond_idx == 0:
                ax.set_ylabel('Autocorrelation')
            ax.grid(True, alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(-0.01, 0.15)
    
    plt.tight_layout()
    plt.show()



'''