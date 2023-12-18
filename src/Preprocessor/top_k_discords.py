# coding: utf-8

import numpy as np
import matrixprofile as mp

import config


def find_discords(matrix_profile, m, discords_num) -> dict:
    """ Return Top-k discords in time series
        
    Find Top-k discords in the time series
    """
    
    discords = mp.discover.discords(matrix_profile, m//2, k=discords_num)
    
    return discords


def construct_discords_annotation(discords, n, m):

    discords_annotation = [1]*n
    anomaly_class = -1
    
    for i in range(len(discords)):
        discords_idxs = np.arange(discords[i]-int(m), discords[i]+int(m))
        discords_idxs = discords_idxs[discords_idxs >= 0]
        discords_idxs = discords_idxs[discords_idxs < n]
        
        for idx in discords_idxs:
            discords_annotation[idx] = anomaly_class

    return discords_annotation
