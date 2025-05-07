#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:32:21 2025

@author: estevao
"""

import os
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import pickle
from scipy import signal, stats
#from scipy import stats

import matplotlib.pyplot as plt





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
#folderPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}/'
#sortData = loadmat(folderPath+f'times_novissimo_data4sorting_{recordingSite}.mat')
#preSortData = loadmat(folderPath+f'data4sorting_{recordingSite}.mat')





# Get selected record
folderPath = f'/home/estevao/Documents/visLab/analise_GLE/data4sorting/spkData_{recordingSite}/'
sortingPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}/'
# adapted to test
sortingPath = f'/home/estevao/Documents/visLab/analise_estevao/fromTakashiPc/jafeitos/spkData_{recordingSite}_otherSort/'

with open(folderPath+f'{record}_data.pkl', 'rb') as f:
    singleData = pickle.load(f)

# Get full recordingSite data (pre-sorting) - to select single record info
#preSortData = loadmat(folderPath+f'data4sorting_{recordingSite}.mat')
# is in different order... so have to use th eolder for now
preSortData = loadmat(sortingPath+f'data4sorting_{recordingSite}.mat')


sortData = loadmat(sortingPath+f'times_novissimo_data4sorting_{recordingSite}.mat')



# -----------------

# 3 - GET DESIRED RECORD INSIDE THE SORTED-DATA (it has them all in a continuous fashion)


# 3a - Search the record

name = preSortData['numberOf_spks_perTrial'][1][1].item()

numPreSpks=[]
for nSpks, recId in preSortData['numberOf_spks_perTrial']:
    print(recId)
    if record in recId:
        print(f'encontrou {recId}')
        selected_nSpks = nSpks
        selected_nSpks_sum = np.sum(selected_nSpks)
        break
        
        # vai para o sorted e faz um slice desta regiao de interesse
        
    else:
        nonSelected = np.sum(nSpks)
        numPreSpks.append(nonSelected)
numPreSpks_sum = np.sum(numPreSpks)

# obs: So I will get up to this value: numPreSpks_sum + selected_nSpks_sum

# 3b - slice the record from inside the sortedData
# SLICE WAVE_CLUS/SORTED DATA
spk_sort = sortData['cluster_class']
slicedSpks = spk_sort[numPreSpks_sum:(numPreSpks_sum + selected_nSpks_sum),:]

# get WFM just to compare with the original data and visually confirm it was sliced correctly
wfm_sort = sortData['spikes']
slicedWfm = wfm_sort[numPreSpks_sum:(numPreSpks_sum + selected_nSpks_sum),:]




# -----------------

# 4 - RE-CONSTRUCT clusters_id IN TRIAL FORMATING

clusId = slicedSpks[:, 0]

clusId_perTrial = []
spkCount = 0
for nSpks in singleData['numSpk_perTrial']:
    #print(nSpks)
    clusId_singleTrl = clusId[spkCount:(spkCount + nSpks)]
    clusId_perTrial.append(clusId_singleTrl)
    
    spkCount += nSpks  # keep the count of how many went by



# -----------------

# 5 - ADD clus_ID INTO SINGLE RECORD DATA (is this really necessary?)

singleData['clusId_perTrial'] = clusId_perTrial
singleData['clusId'] = clusId


# -----------------

# 6 - AUTOCORRELATION - SINGLE SUA ACROSS TRIALS AND CONDITIONS


## from older code... have to adpat

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




# from trial 1
#singleTrl_index = np.array(singleData['index_perTrial'][1])
#singleTrl_clusId = singleData['clusId_perTrial'][0]
#singleTrl_wfm = singleData['spk_perTrial'][1]
#
## get data just from SUA==2 (still just on the first trial)
#index_singleSua = singleTrl_index[singleTrl_clusId==2] / 32
#wfm_singleSua = singleTrl_wfm[singleTrl_clusId==2]
#
#single_lags, single_acorr = autocorrelation_analysis(index_singleSua, bin_size=2, window=250, normalize=True)

# 6a - loop thru each SUA in each trial to get mean autoCorr

## choose SUA
#sua_selected = 3
#
## loop thru each trial to get autoCorr
#all_acorrs = []
#for trlIdx in range(len(singleData['index_perTrial'])):
#    singleTrl_index = np.array(singleData['index_perTrial'][trlIdx + 1])
#    singleTrl_clusId = singleData['clusId_perTrial'][trlIdx]
#    singleTrl_wfm = singleData['spk_perTrial'][trlIdx + 1]
#    
#    # get data just from SUA==2 (still just on the first trial)
#    index_singleSua = singleTrl_index[singleTrl_clusId==sua_selected] / 32
#    wfm_singleSua = singleTrl_wfm[singleTrl_clusId==sua_selected]
#    
#    # AUTO-CORRELATION
#    single_lags, single_acorr = autocorrelation_analysis(
#                                        index_singleSua,
#                                        bin_size=2,
#                                        window=250,
#                                        normalize=True)
#    
#    all_acorrs.append(single_acorr)
#
## get mean autoCorr
#acorr_mean = np.mean(all_acorrs,axis=0)


# LOOP THRU SUAs
#suaIdx = 0
acorr_allSUAs = []
for suaIdx in range(int(np.max(singleData['clusId']))):
   # if suaIdx != 0: # sua zero is the non-clusters spks in wave_clus
        suaIdx = suaIdx +1
        
        
        # LOOP THRU CONDITIONS (diff stimuli)
        acorr_allConds = []
        wfm_allConds = []
        for condIdx in range(np.max(singleData['stimConditions'])):
            condIdx = condIdx+1
        #    condIdx = 2
        
            
            ## trying for each condition now
            
            # cannot do boolean on dict, so key indices
            selected_trial_indices = [
                trial_idx for trial_idx, condition 
                in enumerate(singleData['stimConditions']) 
                if condition == condIdx
            ]
            
            selected_trial_indices = np.array(selected_trial_indices)
            
            # Get just data from this condition
            index_singleCond = {
                trial_idx: singleData['index_perTrial'][trial_idx] 
                for trial_idx in (selected_trial_indices+1)
                                }
            
            clusId_singleCond = {
                trial_idx: singleData['clusId_perTrial'][trial_idx] 
                for trial_idx in selected_trial_indices
                                }
            
            spk_singleCond = {
                trial_idx: singleData['spk_perTrial'][trial_idx] 
                for trial_idx in (selected_trial_indices+1)
                                }
            
            
            
            # 6a - loop thru each SUA in each trial to get mean autoCorr
            
            # choose SUA
        #    suaIdx = 2
            
            # loop thru each trial to get autoCorr          index_singleCond[10]
            all_acorrs = []
            all_wfms = []
            for trlIdx in range(len(index_singleCond)):
                singleTrl_index = np.array(index_singleCond[selected_trial_indices[trlIdx] + 1])
                singleTrl_clusId = clusId_singleCond[selected_trial_indices[trlIdx]]
                singleTrl_wfm = spk_singleCond[selected_trial_indices[trlIdx] + 1]
                
                # get data just from SUA==2 (still just on the first trial)
                index_singleSua = singleTrl_index[singleTrl_clusId==suaIdx] / 32
                wfm_singleSua = singleTrl_wfm[singleTrl_clusId==suaIdx]
                
                # AUTO-CORRELATION
                single_lags, single_acorr = autocorrelation_analysis(
                                                    index_singleSua,
                                                    bin_size=2,
                                                    window=150,
                                                    normalize=True)
                
                all_acorrs.append(single_acorr)
                all_wfms.append(np.mean(wfm_singleSua, axis = 0))
            
            
            
            # get mean autoCorr & wfm
            acorr_mean = np.mean(all_acorrs,axis=0)
            wfm_mean = np.mean(all_wfms,axis=0)
            
            acorr_allConds.append(acorr_mean)
            wfm_allConds.append(wfm_mean)
        
        
        
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # 6b - FIGURE
        
        # Compute time axis
        timePoints = (np.arange(38) / 32000 ) * 1000  # Time in ms  (numSamples/sampleRate * 1000)

        
        fig, axs = plt.subplots(1, len(np.unique(singleData['stimConditions'])), figsize=(15, 4), sharex=True)
        
        # figure title
        fig.suptitle(f'SUA {suaIdx}')#, y=1.05, fontsize=12)
        for cond_idx in range(len(np.unique(singleData['stimConditions']))):
            # get info form this condition|SUA
            acorr = acorr_allConds[cond_idx]
            lags = single_lags

            # subplots for each condition
            if len(lags) > 0:
                axs[cond_idx].plot(lags, acorr, linewidth = 0.7)
                axs[cond_idx].axvline(x=25, color='r', linestyle='--', alpha=0.9, linewidth = 0.7)
    #            if cond_idx == 0:
    #                axs[cond_idx].set_ylabel('Autocorrelation')
                axs[cond_idx].set_title(f'Condition {cond_idx+1}', pad=10)
                axs[cond_idx].grid(True, alpha=0.3)
   #             axs[cond_idx].set_ylim(-0.01, 0.5)
                axs[cond_idx].spines[['top', 'right']].set_visible(False)
                
                # make plot be in the center by mean value of acorr
                axs[cond_idx].set_ylim(np.mean(acorr)-0.05, np.mean(acorr)+0.2)
                
                # WAVEFORM INSET
                #inset = axs[cond_idx].add_axes([0.6, 0.6, 0.3, 0.3])
                inset = inset_axes(axs[cond_idx], 
                          width="25%",  # width of inset
                          height="25%",  # height of inset
                          loc='upper right')
                inset.plot(timePoints, wfm_allConds[cond_idx], linewidth=0.5, color = 'k')
                inset.tick_params(axis='both', which='major', length=2)  # Shorter ticks
                inset.spines[['top', 'right']].set_visible(False)
                for spine in inset.spines.values():
                    spine.set_linewidth(0.25)  # Thinner border
                inset.tick_params(axis='both', which='major', labelsize=5)  # Adjust size
                
                #remove y-axis
                inset.tick_params(axis='y', which='both', left=False, labelleft=False)  # Adjust size
                
        plt.tight_layout()
        plt.show()
        
        
        plt.figure(figsize=(10, 4))
        for cond_idx, wfm in enumerate(wfm_allConds):
            plt.plot(timePoints, wfm, label=f'Cond {cond_idx+1}', alpha=0.7)
        plt.title(f'SUA {suaIdx} - Waveforms Across Conditions')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()



