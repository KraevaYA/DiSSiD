# coding: utf-8

import matrixprofile as mp


def find_mp(ts, m) -> dict:
    """ Return Matrix Profile
    
    Calculate the Matrix Profile
    """
    
    matrix_profile = mp.compute(ts, m//2) 
    
    return matrix_profile
