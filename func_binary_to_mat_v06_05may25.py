#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:07:24 2025

@author: estevao

>i2 instead of >i4
2 indicates 2 bytes per value (16 bits)
The > still indicates big-endian byte order

The pattern is consistent:

>i4 = big-endian 32-bit integer (4 bytes)
>i2 = big-endian 16-bit integer (2 bytes)
>i8 = big-endian 64-bit integer (8 bytes)

Similarly, if you needed little-endian, you would use <i2 instead.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pickle
#import struct
import continuous_ms_time  # estevao qm criou
#import readBinaryFile  # estevao qm criou
from readBinaryFile_v03_05may25 import readBin  # estevao qm criou


'''
# na ordem:

chosenRecord = 'gle04b06' # certo

chosenRecord = 'gle04c08' # certo

chosenRecord = 'gle04c29' # certo

chosenRecord = 'gle04d09' # certo

chosenRecord = 'gle04e03' # certo

# f >> NO m-sequence FOR THIS POSITION!

chosenRecord = 'gle04g09' # certo

chosenRecord = 'gle04h03' # certo

chosenRecord = 'gle04i03' # certo

chosenRecord = 'gle04j26' # certo

chosenRecord = 'gle05c08'  # NAO LOCALIZA ARQUIVO!!
chosenRecord = 'gle0508'  # ERRO NO NOME DO SAVE?!?!  CERTO

chosenRecord = 'gle05d08' # certo

chosenRecord = 'gle04j23' # oFlicker30Hz
chosenRecord = 'gle04e06' # digFlicker40Hz, 50 trls
chosenRecord = 'gle04c29' # certo

chosenRecord = 'gle05o09'#'gle05c09'#'gle04d02'

folderPath = '/home/estevao/Documents/visLab/proj_gle_cat_2025/gle/'+chosenRecord
fileSpk = folderPath+'/'+chosenRecord+'.spike'
fileWfm = folderPath+'/'+chosenRecord+'.swave'

'''


def bin2mat(chosenRecord, folderPath, recordingSite):

    fileSpk = folderPath+'/'+chosenRecord+'.spike'
    fileWfm = folderPath+'/'+chosenRecord+'.swave'
    fileStim = folderPath+'/'+chosenRecord+'.stim'
    fileAna = folderPath+'/'+chosenRecord+'.ana'
    fileBhv = folderPath+'/'+chosenRecord+'.bhv'


    # -----------
    
    # 1) Get DATA (.spike must be read before .swave)


    # 1a) read spk timestamps (sometimes called as index/times/spks etc)
    spk_time_perTrial = readBin(fileSpk)

    # check if single or multiple trials
    if type(spk_time_perTrial) == tuple:
        tot_num_trls = 1  # when 1 trial it runs as tuple an doesNOT creates a dict
        numSpk_perTrial = spk_time_perTrial[0]  # get spks per trial> to find data after concatenating
    else:
        tot_num_trls = len(spk_time_perTrial)  # inserted each trial into a dict
        numSpk_perTrial = [
            len(spk_time_perTrial[ii+1])
            for ii in range(len(spk_time_perTrial))
            ]  # get spks per trial> to find data after concatenating


    # 1b) read spk waveforms
    spk_wfm_perTrial = readBin(fileWfm, tot_num_trls=tot_num_trls)  # must inform num_trls
    
    # 1c) read .stim
    metaData_stim_perTrial = readBin(fileStim)

    # 1d) read .ana
    metaData_ana_perTrial = readBin(fileAna)
    
    # 1e) read .bhv
    metaData_bhv_perTrial = readBin(fileBhv)

    # 1c) read .info (?!)





    # -----------

    # 2) CONCATENATE all SPIKES to run the SPIKESORTING


    
    if tot_num_trls > 1:
        # for spkTiming
        spk_time_concat = sum(spk_time_perTrial.values(), ())
        # for wfm
        spk_wfm_concat = np.vstack(list(spk_wfm_perTrial.values()))
    else:  # if single trial
        # for spkTiming
        spk_time_concat = spk_time_perTrial[1:]  # just remove the 1st element (which counts the num_spks)
        # for wfm
        spk_wfm_concat = spk_wfm_perTrial[1]
    
# previous version
#    spk_wfm = [
#        wfm for inner_dict in spk_wfm_perTrial.values()
#        for wfm in inner_dict['waveforms']
#        ]








    
    
    
    # -----------
    
    # 3) FINAL ARRAYS AND TIME CONVERSION (into ms)
    
    # 3a- wfm
#    array_spkWfm = np.array(spk_wfm)  # completelly unecessary to create another

    # 3b- spk-time (timestamps)
    samplingRate = 32000  # 32kHz  (32000 events recorded per second)
    array_spkTime_in_ms = (np.array(spk_time_concat[:]) / 32000 ) * 1000
    if tot_num_trls > 1:  # if NOT a continuous record (has mult-trls)
        array_spkTime_fakeContinuous_ms = continuous_ms_time.cont(spk_time_perTrial)
    else:
        array_spkTime_fakeContinuous_ms = ['tot_num_trls == 1; thus array not created']
    
    # 3c- bhv
    array_bhv = np.array(metaData_bhv_perTrial)[1:]
    
    # 3d - stim
    array_stim = np.array(metaData_stim_perTrial)[1:]
    
    # 3e- ana
    array_ana = np.array(metaData_ana_perTrial)
    
    
    
    
    
    # -----------
    
    ## 4) FIND PEAK VALUE FOR EACH SPIKE (if data is aligned or not)
    
    peak_all = []
    for spk in range(np.size(spk_wfm_concat,0)):
        peak_position = pd.Series(spk_wfm_concat[spk, :]).idxmax()
        peak_all.append(peak_position)
    
    
    
    
    
    # -----------
    
    # 5) FIGURE PREVIEW
    time_ms = np.linspace(0, (38/32), 38)
    
    # for a 32KiloSample/sec  (32kHz)
    # 1 evt/32000 s == 1evt/32ms
    
    '''
    # FIGURE WFMs preview
    fig, ax = plt.subplots(figsize=(15, 5))
    #for spk in range(len(spk_wfm['waveforms'])):
    for spk in range(100):
        ax = sns.lineplot(x=time_ms, y=spk_wfm_concat[spk, :],
                          linewidth=.6,
                          alpha=.2,
                          color='black')
        #ax.set_xticks(np.linspace(0, 1.25, 10))
        ax.set_xticks(np.arange(0, 1.375, 0.125))
        sns.despine()
    '''
    
    
    
    
    
    # -----------
    
    # 6) SAVE AS MATLAB (.mat) FILE FOR SPIKE SORTING (wave_clus - quiroga, 2018)
    
    
    
    # 6a) create dic for this record's infos
    
    record_dict = {
        #general info
        'record': chosenRecord,
        'time_ms': time_ms,
        'num_of_trls': tot_num_trls,
        'numSpk_perTrial': numSpk_perTrial,

        # timeStamps (spike time)
        'index': array_spkTime_in_ms,
        'index_continuous': array_spkTime_fakeContinuous_ms,
        'index_perTrial': spk_time_perTrial,

        # waveform
        'spk_perTrial': spk_wfm_perTrial,
        'spikes': spk_wfm_concat,

        # metadata
        'analog': array_ana,
        'bhv': array_bhv,
        'stimConditions': array_stim
        }
    
    
    return record_dict
    
    
# copy from the last as is all the same












