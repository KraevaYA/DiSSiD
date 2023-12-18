# coding: utf-8

import pandas as pd
import numpy as np
import math
import stumpy
from stumpy import core
from stumpy.core import check_window_size, _get_mask_slices
from stumpy.mpdist import _mpdist_vect
from stumpy.aampdist_snippets import aampdist_snippets
from sklearn.cluster import KMeans
import itertools
import sys

from config import TOP_K_PROFILES_PER_SNIPPET


def _get_all_profiles(T, m, percentage=1.0, s=None, mpdist_percentage=0.05,
                      mpdist_k=None, mpdist_custom_func=None):
    
    if m > T.shape[0] // 2:  # pragma: no cover
        raise ValueError(f"The window size {m} for each non-overlapping subsequence is too large "
                         f"for a time series with length {T.shape[0]}. "
                         f"Please try `m <= len(T) // 2`.")
    
    right_pad = 0
    if T.shape[0] % m != 0:
        right_pad = int(m * np.ceil(T.shape[0] / m) - T.shape[0])
        pad_width = (0, right_pad)
        T = np.pad(T, pad_width, mode="constant", constant_values=np.nan)

    n_padded = T.shape[0]
    D = np.empty(((n_padded // m) - 1, n_padded - m + 1), dtype=np.float64)

    if s is not None:
        s = min(int(s), m)
    else:
        percentage = np.clip(percentage, 0.0, 1.0)
        s = min(math.ceil(percentage * m), m)

    M_T, Σ_T = stumpy.core.compute_mean_std(T, s)

    # Iterate over non-overlapping subsequences, see Definition 3
    for i in range((n_padded // m) - 1):
        start = i * m
        stop = (i + 1) * m
        S_i = T[start:stop]
        D[i, :] = _mpdist_vect(S_i, T, s, mpdist_percentage, mpdist_k, mpdist_custom_func)
        
    stop_idx = n_padded - m + 1 - right_pad
    D = D[:, :stop_idx]
        
    return D


def calculate_sse(profiles):
    
    sse = sys.float_info.max
    min_profile = [min(index) for index in zip(*profiles)]
    sse = np.sum(min_profile)
    
    return sse


def get_mpdist_profiles(D, combination, n_clusters):
    
    profiles = []
    
    for i in range(n_clusters):
        profiles.append(D[combination[i]])
    
    return profiles


@core.non_normalized(aampdist_snippets)
def find_snippets(T, m, k, percentage=1.0, s=None, mpdist_percentage=0.05,
                  mpdist_k=None, normalize=True, p=2.0):
    
    T = stumpy.core._preprocess(T)
    
    if m > T.shape[0] // 2:  # pragma: no cover
        raise ValueError(f"The snippet window size of {m} is too large for a time series with "
                         f"length {T.shape[0]}. Please try `m <= len(T) // 2`.")

    check_window_size(m, max_size=T.shape[0] // 2)

    # 1. Find all MPdist-profiles
    D = _get_all_profiles(T, m, percentage=percentage, s=s,
                          mpdist_percentage=mpdist_percentage, mpdist_k=mpdist_k)

    # ---------------------

    # 2. Select top-k MPdist-profiles with the minimum area ProfileArea under the curve M
    top_k_profiles_num = k*TOP_K_PROFILES_PER_SNIPPET
  
    if (top_k_profiles_num > D.shape[0]):
        top_k_profiles_num = D.shape[0]

    top_k_profiles = np.empty((top_k_profiles_num, D.shape[-1]), dtype=np.float64)
    Q = np.full(D.shape[-1], np.inf, dtype=np.float64)
    top_k_profiles_idxs = []
    
    for i in range(top_k_profiles_num):
        profile_areas = np.sum(np.minimum(D, Q), axis=1)
        idx = np.argmin(profile_areas)
        top_k_profiles[i] = D[idx]
        Q[:] = np.minimum(D[idx], Q)
        top_k_profiles_idxs.append(idx)

    # 3. Find the differences between each MPdist-profile and all MPdist-profiles from top-k profiles set
    profiles_differences = []
    for i in range(top_k_profiles_num):
        one_profile_differences = []
        for j in range(top_k_profiles_num):
            diff = np.sum(np.abs(np.subtract(top_k_profiles[i], top_k_profiles[j])))
            one_profile_differences.append(diff)
        profiles_differences.append(one_profile_differences)

    # 4. Cluster rows, containing the differences between each MPdist-profile and all MPdist-profiles, using k-means algorithm
    km = KMeans(n_clusters=k, init='k-means++')
    km.fit(np.array(profiles_differences))
    top_k_profiles_labels = km.labels_

    profiles_clusters = []

    for i in range(k):
        cluster_label = i
        profiles_one_cluster = []
        
        for j in range(top_k_profiles_num):
            if ((cluster_label == top_k_profiles_labels[j])):
                profiles_one_cluster.append(j)

        profiles_clusters.append(profiles_one_cluster)

    # 5. Find combinations from multiple sets (clusters which are found on the previous step)
    clusters_combinations = list(itertools.product(*profiles_clusters))

    # 6. Find minimum SSE error
    min_sse = sys.float_info.max
    improved_profiles_combinaion = clusters_combinations[0]

    for i in range(len(clusters_combinations)):
        combination = clusters_combinations[i]
        combination_profiles = get_mpdist_profiles(top_k_profiles, combination, k)
        combination_sse = calculate_sse(combination_profiles)
        
        if (combination_sse < min_sse):
            min_sse = combination_sse
            improved_profiles_combinaion = combination
            #print(i)

    # 7. Get the improved MPdist-profiles and their indexes
    improved_profiles = []
    improved_profiles_idxs = []

    for i in range(k):
        improved_profiles_idxs.append(top_k_profiles_idxs[improved_profiles_combinaion[i]])
        improved_profiles.append(top_k_profiles[improved_profiles_combinaion[i]])

    # 8. Find the improved snippets and characteristics
    pad_width = (0, int(m * np.ceil(T.shape[0] / m) - T.shape[0]))
    n_padded = T.shape[0] + pad_width[1]
    D = np.array(improved_profiles)

    improved_snippets = np.empty((k, m), dtype=np.float64)
    improved_snippets_indices = np.empty(k, dtype=np.int64)
    improved_snippets_profiles = np.empty((k, D.shape[-1]), dtype=np.float64)
    improved_snippets_fractions = np.empty(k, dtype=np.float64)
    improved_snippets_areas = np.empty(k, dtype=np.float64)
    Q = np.full(D.shape[-1], np.inf, dtype=np.float64)
    indices = np.arange(0, n_padded - m, m, dtype=np.int64)
    improved_snippets_regimes_list = []

    for i in range(k):
        profile_areas = np.sum(np.minimum(D, Q), axis=1)
        idx = np.argmin(profile_areas)
        
        improved_snippets[i] = T[indices[improved_profiles_idxs[idx]] : indices[improved_profiles_idxs[idx]] + m]
        improved_snippets_indices[i] = indices[improved_profiles_idxs[idx]]
        improved_snippets_profiles[i] = improved_profiles[idx]
        improved_snippets_areas[i] = np.sum(np.minimum(D[idx], Q))
        Q[:] = np.minimum(D[idx], Q)

    total_min = np.min(improved_snippets_profiles, axis=0)

    for i in range(k):
        mask = improved_snippets_profiles[i] <= total_min
        improved_snippets_fractions[i] = np.sum(mask) / total_min.shape[0]
        total_min = total_min - mask.astype(np.float64)
        slices = _get_mask_slices(mask)
        improved_snippets_regimes_list.append(slices)

    n_slices = [regime.shape[0] for regime in improved_snippets_regimes_list]
    improved_snippets_regimes = np.empty((sum(n_slices), 3), dtype=np.int64)
    improved_snippets_regimes[:, 0] = np.repeat(np.arange(len(improved_snippets_regimes_list)), n_slices)
    improved_snippets_regimes[:, 1:] = np.vstack(improved_snippets_regimes_list)
    
    return (improved_snippets,
            improved_snippets_indices,
            improved_snippets_profiles,
            improved_snippets_fractions,
            improved_snippets_areas,
            improved_snippets_regimes,
            )


def find_mpdist_percentage(m, l) -> float:
    """
    """
    
    default_k_percentage = 0.05

    k_percentage = 1 + (1-2*l)/(2*m-2*l+2)
    k_percentage = max(default_k_percentage, k_percentage)
    
    return k_percentage


def find_snippets_with_optimization(ts, m, l, snippets_num) -> dict:
    
    """ Return snippets in time series
    
    Find improved snippets using the optimization in the time series
    
    """
    
    k_percentage = find_mpdist_percentage(m, l)
    
    improved_snippets = find_snippets(T=ts, m=m, k=snippets_num, s=l, mpdist_percentage=k_percentage)
    
    snippets = {'snippets': improved_snippets[0], 'indices': improved_snippets[1],
        'profiles': improved_snippets[2], 'fractions': improved_snippets[3],
        'areas': improved_snippets[4], 'regimes': improved_snippets[5]}
    
    return snippets


def find_snippets_without_optimization(ts, m, l, snippets_num) -> dict:
    """ Return snippets in time series
        
        Find snippets using the original algorithm without optimization in the time series
        
        Parameters
        ----------
        T : The time series or sequence for which to find the snippets
        m : The snippet length
        l : The snippet mini_length
        k : The percentage of distances that will be used to report 'mpdist'
        snippets_num : The number of snippets
        
        """

    k_percentage = find_mpdist_percentage(m, l)
    
    stumpy_snippets = stumpy.snippets(ts, m, snippets_num, s=l, mpdist_percentage=k_percentage)
    
    snippets = {'snippets': stumpy_snippets[0], 'indices': stumpy_snippets[1],
                'profiles': stumpy_snippets[2], 'fractions': stumpy_snippets[3],
                'areas': stumpy_snippets[4], 'regimes': stumpy_snippets[5]}
    
    return snippets


def find_profiles_curve(profiles, snippets_num) -> list:
    """ Return MPdist-profiles curve M
    
        Find MPdist-profiles curve M
    
        Parameters
        ----------
        profiles : All MPdist-profiles of time series
        snippets_num : The number of snippets

    """
    
    profiles_curve = []
    
    for i in range(len(profiles[0])):
        min_value = float("inf")
        for j in range(snippets_num):
            if (min_value > profiles[j][i]):
                min_value = profiles[j][i]
        profiles_curve.append(min_value)
    
    return profiles_curve


def find_mpdist_regimes(regimes, profiles, snippets_num):
    """
        Parameters
        ----------
        regimes :
        profiles :
        snippets_num : The number of snippets
        
    """
    
    mpdist_regimes = []
    
    for snippet_label in range(snippets_num):
        
        mpdist_regime = []
        mpdist_regime_idx = []
        
        for j in range(len(regimes)):
            regime_label = regimes[j][0]
            if (snippet_label == regime_label):
                start_regime = regimes[j][1]
                end_regime = regimes[j][2]
                mpdist_regime.extend(profiles[snippet_label][start_regime:end_regime])
                mpdist_regime_idx.extend(np.arange(start_regime, end_regime))

        mpdist_regimes.append(pd.Series(mpdist_regime, index=mpdist_regime_idx))

    return mpdist_regimes


def find_regimes_profiles(mpdist_regimes, n, snippets_num):
    """
        Parameters
        ----------
        mpdist_regimes :
        n : The length of MPdist-profile
        snippets_num : The number of snippets
        
    """
    
    regimes_profiles = []
    
    for i in range(snippets_num):
        
        regimes_profiles.append([0]*n)
        
        for j in range(len(mpdist_regimes[i].keys())):
            regimes_profiles[i][mpdist_regimes[i].keys()[j]] = mpdist_regimes[i].values[j]

    return regimes_profiles


def find_mpdist_all_regimes(regimes, profiles, snippets_num):
    """
        Parameters
        ----------
        regimes :
        profiles :
        snippets_num : The number of snippets
        
    """
    
    mpdist_regimes = []
    
    for snippet_label in range(snippets_num):
        
        mpdist_regime = []
        mpdist_regime_idx = []
        
        for j in range(len(regimes)):
            regime_label = regimes[j][0]
            if (snippet_label == regime_label):
                start_regime = regimes[j][1]
                end_regime = regimes[j][2]
                
                for ind in range(start_regime, end_regime, 1):
                    mpdist = []
                    for num_snip in range(snippets_num):
                        mpdist.append(profiles[num_snip][ind])
                    
                    mpdist_regime.extend([mpdist])
                
                mpdist_regime_idx.extend(np.arange(start_regime, end_regime))
        
        mpdist_regimes.append(pd.DataFrame(mpdist_regime, index=mpdist_regime_idx))

    return mpdist_regimes
