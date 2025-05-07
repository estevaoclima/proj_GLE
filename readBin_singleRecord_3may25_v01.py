#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:27:46 2025

@author: estevao
"""


import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import struct
import seaborn as sns

# -----------------

# 1 - Choose Record

record = 'gle05a03'#'gle05c03'#'gle04d02'
record = 'gle03a11'#'gle05c03'#'gle04d02'
record = 'gle04c01'#'gle05c03'#'gle04d02'

# error when counting trials
record = 'gle04c21'#'gle05c03'#'gle04d02'
record = 'gle04c30'#'gle05c03'#'gle04d02'
# from 30 up to 42 !!!!
record = 'gle04c42'#'gle05c03'#'gle04d02'


record = 'gle04g17'#'gle05c03'#'gle04d02'
record = 'gle05o09'#'gle05c09'#'gle04d02'

record = 'gle04d01'  # Num of conditions is Not wcorrect!!!



## counting from the 1st
#error
record = 'gle04c04'

record = 'gle04c14' #not sure?!
record = 'gle04c21' # size & cond error

record = 'gle04c34' # could not find drecord
record = 'gle04c44'# could not find drecord


record = 'gle04f01' # could NOT find it
record = 'gle04f02' #could not find the record
record = 'gle04f09' # wrong number of trials!!
record = 'gle04f10' # wrong number of trials
record = 'gle04f20'  # file not found!



record = 'gle04f23' # possible error
record = 'gle04f24' # possible erroe
record = 'gle04f25' # possible error
record = 'gle04f27'  # file not found


#coorect (?)
record = 'gle04b05'#'gle05c09'#'gle04d02'
record = 'gle04c01'#'gle05c09'#'gle04d02'
record = 'gle04c02'#'gle05c09'#'gle04d02'
record = 'gle04c03'
record = 'gle04c05'
record = 'gle04c06'
record = 'gle04c07'
record = 'gle04c09'
record = 'gle04c10'
record = 'gle04c11'
record = 'gle04c12'
record = 'gle04c13'
record = 'gle04c15'
record = 'gle04c16'
record = 'gle04c17'
record = 'gle04c18'
record = 'gle04c19'
record = 'gle04c20'
record = 'gle04c22'  # GREAT EX of error! #21 was wrong
record = 'gle04c23'
record = 'gle04c24'
record = 'gle04c25'
record = 'gle04c26'
record = 'gle04c27'
record = 'gle04c28'
record = 'gle04c29'
record = 'gle04c30'
record = 'gle04c31'
record = 'gle04c32'

record = 'gle04c33'

record = 'gle04c35'
record = 'gle04c36'
record = 'gle04c37'
record = 'gle04c38'
record = 'gle04c39'
record = 'gle04c40'
record = 'gle04c41'
record = 'gle04c42'
record = 'gle04c43'
record = 'gle04c44'
record = 'gle04c45'
record = 'gle04c46'
record = 'gle04c47'
record = 'gle04c48'
record = 'gle04c49'
record = 'gle04c50'
record = 'gle04c51'
record = 'gle04c52'
record = 'gle04c53'
record = 'gle04c54'
record = 'gle04c55'
record = 'gle04c56'

record = 'gle04d02'
record = 'gle04d03'
record = 'gle04d04'
record = 'gle04d05'
record = 'gle04d06'
record = 'gle04d07'
record = 'gle04d01'
record = 'gle04d08'
record = 'gle04d09'
record = 'gle04d10'

record = 'gle04e01'
record = 'gle04e02'
record = 'gle04e03'
record = 'gle04e04'
record = 'gle04e05'
record = 'gle04e06'
record = 'gle04e07'
record = 'gle04e08'


record = 'gle04f03'
record = 'gle04f04'
record = 'gle04f05'
record = 'gle04f06'
record = 'gle04f07'
record = 'gle04f08'  # jeroma is the num codn right? trials is ok.
record = 'gle04f11'
record = 'gle04f12'
record = 'gle04f13'
record = 'gle04f14'
record = 'gle04f15'  # flicker jerome ok!
record = 'gle04f16'
record = 'gle04f17'
record = 'gle04f18'
record = 'gle04f19'
record = 'gle04f21'
record = 'gle04f22'
record = 'gle04f26'
record = 'gle04f28'


record = 'gle04g01'
record = 'gle04g02'
record = 'gle04g03'
record = 'gle04g04'
record = 'gle04g05'
record = 'gle04g06'
record = 'gle04g07'
record = 'gle04g08'
record = 'gle04g09'
record = 'gle04g10'
record = 'gle04g11'
record = 'gle04g12'
record = 'gle04g13'
record = 'gle04g14'
record = 'gle04g15'
record = 'gle04g16'
record = 'gle04g17'


record = 'gle04h01'
record = 'gle04h02'
record = 'gle04h03'
record = 'gle04h04'
record = 'gle04h05'
record = 'gle04h06'
record = 'gle04h07'
record = 'gle04h08'
record = 'gle04h09'
record = 'gle04h10'



record = 'gle04i01'
record = 'gle04i02'
record = 'gle04i03'
record = 'gle04i04'
record = 'gle04i05'
record = 'gle04i06'
record = 'gle04i07'
record = 'gle04i08'
record = 'gle04i09'
record = 'gle04i10'

record = 'gle04i11'


record = 'gle04e06'



# -----------------

# 2 - LOAD DATA

recordingSite = record[:-2]

searchPath = '/home/estevao/Documents/visLab/proj_gle_cat_2025/gle/'

folderPath = searchPath+record
os.chdir(folderPath)

chosenRecord = record

fileSpk = folderPath+'/'+chosenRecord+'.spike'






# -----------

# 3) Get DATA (from .spike & .swave - must be in this order!)




# -----------

# 3a) Function used if there is more than one trial on the file
def createDict_4_multTrial(spk_time):
    '''Create dictionary if dataFIle is composed from multiple TRIALS data.'''

    spk_trl_dict = {}
    startingFrom = 0
    trl=0
    while (trl + startingFrom) < len(spk_time):
        firstElement = trl + startingFrom  # primeiro valor
        
        trl_size = spk_time[firstElement]
        
        # check for error
        if firstElement + trl_size +1 > len(spk_time):
            print(f"CAUTION! Incomplete trial data at {trl+1}")
        
        
        spk_trl_dict[trl+1] = spk_time[firstElement+1:(startingFrom + trl_size + trl + 1)]
        
        startingFrom = trl_size + startingFrom
        trl +=1
    return spk_trl_dict
'''
# the following is a chatGPT suggestion that could help if some value is missing and drifts the aligment through all the data
    spk_trl_dict = {}
    pos = 0
    trl = 1
    while pos < len(spk_time):
        if pos + 1 > len(spk_time):
            print(f"⚠️ Trial {trl}: Missing size field.")
            break
        trl_size = spk_time[pos]
        if pos + trl_size + 1 > len(spk_time):
            print(f"⚠️ Trial {trl}: Incomplete (expected {trl_size} spikes).")
            break
        spk_trl_dict[trl] = spk_time[pos + 1: pos + 1 + trl_size]
        pos += 1 + trl_size
        trl += 1
    return spk_trl_dict

'''










# -----------

# 3b) read binary file

filename = fileSpk
"""Get info from binary files knowing it may have int32 or int16 info.

    Get .spike file knowning it has only int32 values.

        '>{fileSize_values}i' means:
            > - Use big-endian byte order
            {fileSize_values} - Repeat the next format character this many times
            i - Interpret each 4-byte chunk as a signed 32-bit integer
        """
with open(filename, 'rb') as f:
    spkData = f.read()  # read the entire file
    fileSize_bytes = len(spkData)  # size in BYTES
    fileSize_values = fileSize_bytes // 4  # as I know it is int32 (1 values==4 bytes )
#            fileSize_values_16 = fileSize_bytes // 2#just to check if it was i16  # as I know it is int32 (1 values==4 bytes )
    spikeTime = struct.unpack(f'>{fileSize_values}i', spkData)
            # check if isnt int16
            #fileSize_values_16 = fileSize_bytes // 2 
#            spikeTime_16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16

    # check if recoding has multiple TRIALS
    if (spikeTime[0] + 1) < len(spikeTime):  # plus_1 because it would not count the first
        spk_dic = createDict_4_multTrial(spikeTime)  # create a dic separating the trials




## for reading other formats

# -----------
# .stim

fileSpk = folderPath+'/'+chosenRecord+'.stim'
filename = fileSpk
"""Get info from binary files knowing it may have int32 or int16 info.

Get .stim file knowning it has only int32 values (info from spass2field fieldtrip).

ABOUT: stim seems to have the 1st value as total num of trials
            and the following are the condition labels

'>{fileSize_values}i' means:
    > - Use big-endian byte order
    {fileSize_values} - Repeat the next format character this many times
    i - Interpret each 4-byte chunk as a signed 32-bit integer

"""
with open(filename, 'rb') as f:
    spkData = f.read()  # read the entire file
    fileSize_bytes = len(spkData)  # size in BYTES
    fileSize_values = fileSize_bytes // 4#4  # as I know it is int32 (1 values==4 bytes )
#            fileSize_values_16 = fileSize_bytes // 2#just to check if it was i16  # as I know it is int32 (1 values==4 bytes )
    metadata_stim = struct.unpack(f'>{fileSize_values}i', spkData)
            # check if isnt int16
            #fileSize_values_16 = fileSize_bytes // 2 
#            spikeTime_16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16




# -----------
# .stm

fileSpk = folderPath+'/'+chosenRecord+'.stm'
filename = fileSpk
"""Get info from binary files knowing it may have int32 or int16 info.

Get .stm file knowning it has only int32 values (info from spass2field fieldtrip).

ABOUT: stm seems to have the 1st value as total num of trials
            and the following are the condition labels

'>{fileSize_values}i' means:
    > - Use big-endian byte order
    {fileSize_values} - Repeat the next format character this many times
    i - Interpret each 4-byte chunk as a signed 32-bit integer

"""
with open(filename, 'rb') as f:
    spkData = f.read()  # read the entire file
    fileSize_bytes = len(spkData)  # size in BYTES
    fileSize_values = fileSize_bytes // 4#4  # as I know it is int32 (1 values==4 bytes )
#            fileSize_values_16 = fileSize_bytes // 2#just to check if it was i16  # as I know it is int32 (1 values==4 bytes )
    metadata_stm = struct.unpack(f'>{fileSize_values}i', spkData)
            # check if isnt int16
            #fileSize_values_16 = fileSize_bytes // 2 
#            spikeTime_16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16




# -----------
# .ana

fileSpk = folderPath+'/'+chosenRecord+'.ana'
filename = fileSpk
"""Get info from binary files knowing it may have int32 or int16 info.

Get .ana file knowning it has only int16 values (info from spass2field fieldtrip)

ABOUT: ana seems to have the flp data(? == analog?)

'>{fileSize_values}i' means:
    > - Use big-endian byte order
    {fileSize_values} - Repeat the next format character this many times
    i - Interpret each 4-byte chunk as a signed 32-bit integer

        """
with open(filename, 'rb') as f:
    spkData = f.read()  # read the entire file
    fileSize_bytes = len(spkData)  # size in BYTES
    #fileSize_values = fileSize_bytes // 4#4  # as I know it is int32 (1 values==4 bytes )
    fileSize_values_16 = fileSize_bytes // 2#just to check if it was i16  # as I know it is int32 (1 values==4 bytes )
    #metadata_ana = struct.unpack(f'>{fileSize_values}i', spkData)
            # check if isnt int16
            #fileSize_values_16 = fileSize_bytes // 2 
    metadata_ana_i16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16

metadata_ana_i16_volts = np.array(metadata_ana_i16)/32  # 32kHz is that right??

'''
fig, ax = plt.subplots(figsize=(15, 5))
ax = sns.lineplot(np.array(metadata_ana_i16_volts),
                  linewidth=.2,
                  alpha=.9,
                  color='black')
sns.despine()
'''






# -----------
# .bhv

fileSpk = folderPath+'/'+chosenRecord+'.bhv'
filename = fileSpk
"""Get info from binary files knowing it may have int32 or int16 info.

Get .bhv file knowning it has only int32 values (informed by spass2field fieldtrip)

ABOUT: bhv seems to have the 1st value as total num of trials
            and the following are the behaviors

'>{fileSize_values}i' means:
    > - Use big-endian byte order
    {fileSize_values} - Repeat the next format character this many times
    i - Interpret each 4-byte chunk as a signed 32-bit integer

"""
with open(filename, 'rb') as f:
    spkData = f.read()  # read the entire file
    fileSize_bytes = len(spkData)  # size in BYTES
    fileSize_values = fileSize_bytes // 4#4  # as I know it is int32 (1 values==4 bytes )
    metadata_bhv = struct.unpack(f'>{fileSize_values}i', spkData)
            # check if isnt int16
            #fileSize_values_16 = fileSize_bytes // 2 #just to check if it was i16  # as I know it is int32 (1 values==4 bytes )
#            spikeTime_16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16





# -----------
# .info
#  stilll don't how to open it... 




# -----------
# .swave

fileSpk = folderPath+'/'+chosenRecord+'.swave'
filename = fileSpk

"""
Get .swave file knowing it has:
- Header-like metadata with two int32 values
- Followed by values as int16 waveforms
"""

tot_num_trls = len(spk_dic)  #len(spk_time_perTrial)



with open(filename, 'rb') as f:
    
    swave={}
    trls = 0  # keep count of number of trials that should read
    while trls < tot_num_trls:
        result = {}
        waveforms = []
        # Read header (two 32-bit integers)
        header_data = f.read(2 * 4) # 2 elements of 4 bytes each (4 bytes per value (as it is i32))
        number_of_spikes, size_of_waveform = struct.unpack('>ii', header_data)

        result['number_of_spikes'] = number_of_spikes
        result['size_of_waveform'] = size_of_waveform

        # Read waveform data for each spike (all in int16bit)
        spikes_read = 0
        while spikes_read < number_of_spikes:
        #while True:# Read 38 values as 16-bit signed integers (2bytes per value *38 =76 bytes total wfm)
            waveform_data = f.read(size_of_waveform * 2)  # 2 bytes per value

            if len(waveform_data) < size_of_waveform * 2:
                break  # No more complete waveforms

            # Unpack the data - '>' for big-endian, 'h' for 16-bit signed integer
            # Repeat 'h' size_of_waveform times
            format_string = '>' + 'h' * size_of_waveform
            waveform = struct.unpack(format_string, waveform_data)
            waveforms.append(waveform)
            spikes_read += 1

        # Store results
        tot_spks_perTrial = spikes_read  # from the last read trl (I think)
        result['waveforms'] = waveforms  # from last read trl
        trls += 1
        swave[trls] = result  # append this last trial info into a dict with all trls infos







## GROUP DATA AS A DATACLASS (OBJECT)



from dataclasses import dataclass


@dataclass
class DataObj:
    """Store data in a FieldTrip-compatible format."""
    record: record
    index: spk_dic#np.ndarray           # Spike times (ms)
    spikes: swave#np.ndarray          # Waveforms [spikes × samples]
    #time_ms: np.ndarray         # Time axis
    num_of_trls: len(spk_dic)  #int            # Total trials
    #numSpk_perTrial: np.ndarray # Spike counts per trial
    analog: metadata_ana_i16_volts #np.ndarray          # Analog signals [channels × time]
    bhv: metadata_bhv #np.ndarray            # Behavior data
    stimConditions: metadata_stim#np.ndarray  # Stimulus conditions

    # Add a method to convert to FieldTrip format
    def to_fieldtrip(self) -> dict:
        return {
            'trial': [self.spikes.T],      # FieldTrip expects [channels × time]
            'time': [self.time_ms / 1000], # Convert ms to seconds
            'label': ['spike_unit1'],      # Channel names
            'fsample': 30000,              # Example sampling rate
        }


data = DataObj(
     record,
     spk_dic,#np.ndarray           # Spike times (ms)
     swave,#np.ndarray          # Waveforms [spikes × samples]
    #time_ms: np.ndarray         # Time axis
     len(spk_dic),  #int            # Total trials
    #numSpk_perTrial: np.ndarray # Spike counts per trial
     metadata_ana_i16_volts, #np.ndarray          # Analog signals [channels × time]
     metadata_bhv, #np.ndarray            # Behavior data
     metadata_stim#np.ndarray  # Stimulus conditions
)


print(f'\nrecord: {data.record}')
print(f'numConditions: {(np.max(np.array(data.stimConditions[1:])))}')
print(f'numTrls: {data.num_of_trls}')















