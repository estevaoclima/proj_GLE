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
from readBinaryFile_v02_30ab25 import readBin  # estevao qm criou


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
    
    # 1) Get DATA (from .spike & .swave - must be in this order!)
    
    # 1a) read spk timestamps
    spk_time_perTrial = readBin(fileSpk)
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
    
    # 1c) read .stm
    metaData_stim_perTrial = readBin(fileStim)

    # 1d) read .ana
    metaData_ana_perTrial = readBin(fileAna)
    
    # 1e) read .bhv
    metaData_bhv_perTrial = readBin(fileBhv)

    # 1c) read .info (?!)
    
    
    # -----------
    
    # 2) CONCATENATE all SPIKES to run the SPIKESORTING
    
    # for spkTiming
    if tot_num_trls > 1:
        spk_time = sum(spk_time_perTrial.values(), ())
    else:
        spk_time = spk_time_perTrial[1:]  # just remove the 1st element (which counts the num_spks)
    # for wfm
    spk_wfm = [
        wfm for inner_dict in spk_wfm_perTrial.values()
        for wfm in inner_dict['waveforms']
        ]
    
    
    
    
    
    
    
    
    
    # -----------
    
    # 3) FINAL ARRAYS AND TIME CONVERSION (into ms)
    
    # 3a- wfm
    array_spkWfm = np.array(spk_wfm)

    # 3b- spk-time (timestamps)
    samplingRate = 32000  # 32kHz  (32000 events recorded per second)
    array_spkTime_in_ms = np.array(spk_time[:]) / 32000 * 1000
    if tot_num_trls > 1:  # if NOT a continuous record (has mult-trls)
        array_spkTime_continuous_ms = continuous_ms_time.cont(spk_time_perTrial)
    else:
        array_spkTime_continuous_ms = ['tot_num_trls == 1; thus array not created']
    
    # 3c- bhv
    array_bhv = np.array(metaData_bhv_perTrial)[1:]
    
    # 3d - stim
    array_stim = np.array(metaData_stim_perTrial)[1:]
    
    # 3e- ana
    array_ana = np.array(metaData_ana_perTrial)
    
    
    
    
    
    
    # -----------
    
    ## 4) FIND PEAK VALUE FOR EACH SPIKE (if data is aligned or not)
    
    peak_all = []
    for spk in range(np.size(array_spkWfm,0)):
        peak_position = pd.Series(array_spkWfm[spk, :]).idxmax()
        peak_all.append(peak_position)
    
    
    
    
    
    
    # -----------
    
    # 5) FIGURE PREVIEW
    time_ms = np.linspace(0, (38/32), 38)
    '''
    df_wfm = pd.DataFrame(spk_wfm)
    #wfm_array = np.array(spk_wfm['waveforms'])
    wfm_array = np.array(spk_wfm)
    
    # for a 32KiloSample/sec
    # equivalent to 32kHz
    # 1 evt/32000 s == 1evt/32ms
    
    
    # cria figura com os wfms
    fig, ax = plt.subplots(figsize=(15, 5))
    #for spk in range(len(spk_wfm['waveforms'])):
    for spk in range(100):
        ax = sns.lineplot(x=time_ms, y=wfm_array[spk, :],
                          linewidth=.6,
                          alpha=.2,
                          color='black')
        #ax.set_xticks(np.linspace(0, 1.25, 10))
        ax.set_xticks(np.arange(0, 1.375, 0.125))
        sns.despine()
    
    spk_dict = {
           'spk_time': spk_time,  # 1st element was already removed previously
    
           'wfm': spk_wfm
           }
    spk_df = pd.DataFrame(spk_dict)
    '''
    
    
    
    
    
    # -----------
    
    # 6) SAVE AS MATLAB (.mat) FILE FOR SPIKE SORTING (wave_clus - quiroga, 2018)
    
    
    
    # 6a) create dic for this record's infos
    
    record_dict = {
        'record': chosenRecord,
        'index': array_spkTime_in_ms,
        'spikes': array_spkWfm,
        'time_ms': time_ms,
        'num_of_trls': tot_num_trls,
        'numSpk_perTrial': numSpk_perTrial,
        'index_continuous': array_spkTime_continuous_ms,
        'analog': array_ana,
        'bhv': array_bhv,
        'stimConditions': array_stim
        }
    
    
    
    
    
    # SAVE
    from scipy.io import savemat
    import os
    
    outputPath = f'/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/novo_30abr25/data_4_sorting/spkData_{recordingSite}/'
    os.makedirs(outputPath, exist_ok=True)  # create dir; false=prevents creating a copy/overwritting
    
    
    
    # save on matlab format
#    savemat(
#            outputPath+f'{chosenRecord}_data4matlab.mat',
#            record_dict
#            )
    # save on python format (numpy format for numerical arrays)
    #    np.savez(
#                outputPath+f'{chosenRecord}_data4matlab.npz',
#                # files below
#                record_dict
##                )
    
    
    print(chosenRecord)
    print(f'file: {chosenRecord}')
    print(f'length of spk_time: {len(spk_time)}')
    print(f'length of waveformData: {len(spk_wfm)}')
    print(f'total number of trials: {tot_num_trls}')
    #print(f'total number of conditions: {max(np.array(metaData_stm_perTrial)[1:])}')

"""
    # SAVE
    from scipy.io import savemat
    import os
    
    #outputPath = '/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting'+f'/spkData_{chosenRecord}/'
    outputPath = f'/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting/spkData_{recordingSite}/'
    os.makedirs(outputPath, exist_ok=True)  # create dir; false=prevents creating a copy/overwritting
    
#    outputPath_py = f'/home/estevao/Documents/visLab/analise_estevao/handleData4sorting_estevao/data_4_sorting/spkData_{recordingSite}/'
#    os.makedirs(outputPath_py, exist_ok=True)  # create dir; false=prevents creating a copy/overwritting
    
    
    
    # save on matlab format
#    savemat(
            outputPath+f'{chosenRecord}_data4matlab.mat', {
                'index': array_spkTime_in_ms,
                'spikes': array_spkWfm,
                'time_ms': time_ms,
                'num_of_trls': tot_num_trls,
                'numSpk_perTrial': numSpk_perTrial,
                'index_continuous': array_spkTime_continuous_ms,
                'analog': array_ana,
                'bhv': array_bhv,
                'stimConditions': array_stim,
                'record': chosenRecord
                })
    
    # save on python format (numpy format for numerical arrays)
#    np.savez(
            outputPath+f'{chosenRecord}_data4matlab.npz',
            # files below
            index=array_spkTime_in_ms,
            spikes=array_spkWfm,
            time_ms=time_ms,
            num_of_trls=tot_num_trls,
            index_continuous=array_spkTime_continuous_ms,
            numSpk_perTrial=numSpk_perTrial,
            stimConditions=array_stim,
            analog=array_ana,
            bhv=array_bhv,
            record=chosenRecord
            )
"""
    # or save as individual files
    #savemat('index.mat', {'spk_time': array_spkTime_in_ms})
    #savemat('spikes.mat', {'spk_wfm': array_spkWfm})
    #savemat('time_ms.mat', {'time_ms': time_ms})
    





# jsut testing...
chosenRecord = 'gle05o09'#'gle05c09'#'gle04d02'
folderPath = '/home/estevao/Documents/visLab/proj_gle_cat_2025/gle/'+chosenRecord
aasd = bin2mat(chosenRecord, folderPath, recordingSite=0)



##              -----   HAVING A LOOK AT THE FILE   ------
### HEX REPRESENTATION
## OPEn file to understand its structure
#with open(fileSpk, 'rb') as f:
#    first_chunk = f.read(20)  # Read first 100 bytes
#    
#print("Hex representation:")
#print(' '.join(f'{b:02x}' for b in first_chunk))  # Cleaner hex view
#
#
## read as on python 
#open_binData = open(fileSpk, 'rb')
#binData = open_binData.read()
#print(binData[:20])


#              -----   GETING DATA (convert binary into int16 or int32)   ------

#  ----------------------------------------

##         READING THE FILE .spike

#  ----------------------------------------
#if '.spike' in file:
    #with open(filePath + '/' + 'gle05d11.spike', 'rb') as file:
    #    data = pickle.load(file)
    #    print(data)
    
    # depending of data type
    #data32 = np.fromfile(filePath, dtype=np.int32)
    #data16 = np.fromfile(filePath, dtype=np.int16)
#dataSpike = np.fromfile(fileSpk, dtype='>i4')  # i32+big-endian format (>i in struct), where each 4 bytes represent one int32 value

