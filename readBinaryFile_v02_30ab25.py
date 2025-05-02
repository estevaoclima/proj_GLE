#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get spike-time OR wfm from binary file

Ouput: dict
spk-time: binary file with int32 values
spk-wfm: mixed int32 & int16:
    *header-like: with two int32 values:
        number of spk on this specific trial
        number of bins for each wfm (38 bin)
    *waveform for each spk int16 values
for wfm: must inform number of trls

sampleRate: 32kHz


'>{fileSize_values}i' means:
> - Use big-endian byte order
{fileSize_values} - Repeat the next format character this many times
i - Interpret each 4-byte chunk as a signed 32-bit integer


Created on Sat Mar 29 14:00:49 2025
@author: estevaoCLima
"""
import struct

def readBin(filename, tot_num_trls=None):
    """Get info from binary files knowing it may have int32 or int16 info.
    
    tot_num_trls: needed just for wfm (& create by spk-time)
    """
    if '.spike' in filename:
        """
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
#            spikeTime_16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16
            
            # check if recoding has multiple TRIALS
            if (spikeTime[0] + 1) < len(spikeTime):  # plus_1 because it would not count the first
                spk_dic = createDict_4_multTrial(spikeTime)
                return spk_dic # create a dic separating the trials
            else:
                return spikeTime
            # OBS: if sure that file has NO mixed data types (int32 with i16), may use:
                #data = np.fromfile(fileSpk, dtype='>i4')  # i32+big-endian format (>i in struct), where each 4 bytes represent one int32 value


    elif '.swave' in filename:
        """
        Get .swave file knowing it has:
        - Header with two int32 values
        - Following values as int16 waveforms
        """
        
        # check if tot_num_trls is provided
        if tot_num_trls is None:
            raise ValueError("tot_num_trls do Estevao: FALTA  ! RODAR spk-time antes!")
            
        with open(filename, 'rb') as f:
            
            qwe={}
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
                tot_spks_perTrial = spikes_read
                result['waveforms'] = waveforms
                trls += 1
                qwe[trls] = result
        return qwe




# -----------
# .stim

    elif '.stim' in filename:
        """
        Get .stim file knowning it has only int32 values (info from spass2field fieldtrip).

        ABOUT: stim seems to have the 1st value as total num of trials
                    and the following are the condition labels
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
        return metadata_stim




# -----------
# .ana

    elif '.ana' in filename:
        """
        Get .ana file knowning it has only int16 values (info from spass2field fieldtrip)

        ABOUT: ana seems to have the flp data(? == analog?)
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
            return metadata_ana_i16






# -----------
# .bhv

    elif '.bhv' in filename:
        """
        Get .bhv file knowning it has only int32 values (informed by spass2field fieldtrip)

        ABOUT: bhv seems to have the 1st value as total num of trials
                    and the following are the behaviors
        """
        with open(filename, 'rb') as f:
            spkData = f.read()  # read the entire file
            fileSize_bytes = len(spkData)  # size in BYTES
            fileSize_values = fileSize_bytes // 4  #4  # as I know it is int32 (1 values==4 bytes )
            metadata_bhv = struct.unpack(f'>{fileSize_values}i', spkData)
                    # check if isnt int16
                    #fileSize_values_16 = fileSize_bytes // 2 #just to check if it was i16  # as I know it is int32 (1 values==4 bytes )
        #            spikeTime_16 = struct.unpack(f'>{fileSize_values_16}h', spkData)#just to check if was i16
            return metadata_bhv






# -----------
# .info
#  stilll don't how to open it... 







# obs if dataFile is composed of many TRIALS
# 1st is always the number os spks?
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
