#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert trials into a continuous-like space.

Used because could (NOT sure about that!) give a better sorting.
Not sure if that is necessary for sorting using wave_clus-quiroga-2018

# 1000 ms added to create a gap btw trials (to avoid ISI counting as near when change from trial to trial)

Created on Sat Mar 29 13:47:44 2025
@author: estevao
"""
import numpy as np


def cont(spk_time_perTrial):
    """Convert trls into a continuous-like space (better sorting?)."""
    last_trl_endTime = 0
    spkTime_continuous_ms = []

    for trl_idx in spk_time_perTrial.keys():
        trl_spk = (
            (np.array(spk_time_perTrial[trl_idx]) / 32000 * 1000) + last_trl_endTime)

        spkTime_continuous_ms.append(trl_spk)
        last_trl_endTime = trl_spk[-1] + 1000  # 1000 ms added to create a gap btw trials (to avoid ISI counting as near when change from trial to trial)

    single_continuous_ms = np.concatenate(spkTime_continuous_ms)
    return single_continuous_ms








def cont_ORIGINALcomERRO(spk_time_perTrial):
    """Convert trls into a continuous-like space (better sorting?)."""
    last_trl_endTime = 0
    spkTime_continuous_ms = []

    for trl_idx in spk_time_perTrial.keys():
        trl_spk = (
            (np.array(spk_time_perTrial[1]) / 32000 * 1000) + last_trl_endTime)
        trl_spk = (
            (np.array(spk_time_perTrial[trl_idx]) / 32000 * 1000) + last_trl_endTime)
        spkTime_continuous_ms.append(trl_spk)
        last_trl_endTime = trl_spk[-1]

    single_continuous_ms = np.concatenate(spkTime_continuous_ms)
    return single_continuous_ms

