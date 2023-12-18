import math
import numpy as np
import pandas as pd
from vus.models.feature import Window
from vus.metrics import get_metrics
from sklearn.preprocessing import MinMaxScaler

def round_dict_values(d, k):
    return {key: float(f"{value:.{k}f}") for key, value in d.items()}

def scoring(score, labels, slidingWindow):
    # Score normalization
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    
    results = get_metrics(score, labels, metric='all', slidingWindow=slidingWindow) # default metric='vus'
    
    for metric in results.keys():
        print(metric, ':', results[metric])
    
    k = 4
    results = round_dict_values(results, k)

    return results
