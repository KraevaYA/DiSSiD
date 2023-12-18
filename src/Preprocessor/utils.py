import numpy as np
import pandas as pd
import random
import csv
import os
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import config


def parse_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-files', '--input_files', action='store', dest='input_files', nargs="+", help='Input files names', required=True)
    parser.add_argument('-train_lengths', '--train_lengths', action='store', dest='train_lengths', nargs="+", type=int, help='Lengths of input time series', required=True)
    parser.add_argument('-m', '--anomaly_length', action='store', dest='m', type=int, help='Anomaly length (snippet and discord length)', required=True)
    parser.add_argument('-l', '--mini_snippet_length', action='store', dest='l', type=int, help='Mini length for snippet finding based on MPdist measure', required=True)
    parser.add_argument('-alpha', '--discords_fraction', action='store', dest='alpha', default=config.ALPHA, type=float, help='Discord fraction in the time series')
    parser.add_argument('-snippets_num', '--snippets_num', action='store', dest='snippets_num', type=int, help='Number of snippets in the time series', required=True)
    parser.add_argument('-test_normal_set_size', '--test_normal_set_size', action='store', dest='test_normal_set_size', default=config.TEST_NORMAL_SET_SIZE, type=float, help='Fraction of normal samples for test set')
    
    return parser.parse_args()


def create_directory(directory_path):
    
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path


def load_ts(file_path) -> np.array:
    """ Return time series
        
        Load time series from txt file
    """
    
    input_file = open(file_path, "r")
    
    ts = input_file.read().split("\n")[:-1]
    ts = np.array([float(elem) for elem in ts])
    
    return ts


def load_dataset(file_path):
    
    df = pd.read_csv(file_path, header=None).dropna().to_numpy() #sep=" "

    ts = df[:,0].astype(float)
    label = df[:,1].astype(int)

    return ts, label


def split_ts(input_files, train_lengths):

    train_ts = []
    test_ts = []
    train_label = []
    test_label = []
    
    ts_lengths = []
    
    for i in range(len(input_files)):
        ts_path = os.path.join(config.ORIGINAL_TS_DIR, input_files[i])
        ts, label = load_dataset(ts_path)
        ts_lengths.append(len(ts))
        train_ts.extend(ts[:train_lengths[i]])
        test_ts.extend(ts[train_lengths[i]:])
        train_label.extend(label[:train_lengths[i]])
        test_label.extend(label[train_lengths[i]:])

    return train_ts, test_ts, train_label, test_label, ts_lengths


def moving_max(ts, w = 3):

    moving_max_ts = []
    ts_len = len(ts)

    for i in range(w+1):
        moving_max_ts.append(np.max(ts[0:i+w]))
    for i in range(w+1,ts_len-w+1):
        moving_max_ts.append(np.max(ts[i-w:i+w]))
    for i in range(ts_len-w, ts_len):
        moving_max_ts.append(np.max(ts[i:ts_len]))

    return moving_max_ts


def moving_min(ts, w = 3):
    
    moving_min_ts = []
    ts_len = len(ts)
    
    for i in range(ts_len-w+1):
        moving_min_ts.append(np.min(ts[i:i+w]))
    for i in range(ts_len-w, ts_len):
        moving_min_ts.append(np.min(ts[i:ts_len]))
    
    return moving_min_ts


def moving_mean(ts, w = 3):

    moving_mean_ts = []
    ts_len = len(ts)
    
    for i in range(ts_len-w+1):
        moving_mean_ts.append(np.mean(ts[i:i+w]))
    for i in range(ts_len-w, ts_len):
        moving_mean_ts.append(np.mean(ts[i:ts_len]))
    
    return moving_mean_ts


def moving_std(ts, w = 3):
    
    moving_std_ts = []
    ts_len = len(ts)
    
    for i in range(ts_len-w+1):
        moving_std_ts.append(np.std(ts[i:i+w]))
    for i in range(ts_len-w, ts_len):
        moving_std_ts.append(np.std(ts[i:ts_len]))
    
    return moving_std_ts


# the anomaly snippets, their nearest neighbors and discords are labeled the same (abnormal) class = -1
def label_subsequences(snippets, anomalies_annotation, N, m, snippets_num):
    
    normal_label = 0
    anomaly_label = -1
    
    subs_labels = [normal_label]*N
    snippets_fractions = snippets['fractions'].tolist()
    snippets_regimes = np.array(snippets['regimes'])
    
    normal_labels = sum(frac > config.SNIPPETS_FRACTION_THRESHOLD for frac in snippets_fractions)
    
    # label the anomaly snippets and its nearest neighbors whose the fractions are less than snippets_fraction_threshold
    for i in range(snippets_num):
        slices_of_indices = snippets_regimes[np.where(snippets_regimes[:,0]==i)][:,1:]
        
        if (snippets_fractions[i] > config.SNIPPETS_FRACTION_THRESHOLD):
            current_label = normal_label
            normal_label = normal_label + 1
        else:
            current_label = anomaly_label
    
        for per_slice in slices_of_indices:
            start_idx = per_slice[0]
            stop_idx = per_slice[1]
            
            for j in range(stop_idx-start_idx):
                #if (start_idx+j != N):
                subs_labels[start_idx+j] = current_label

    # Label the anomalous subsequences which are discords or snippets anomalies
    subs_labels = np.array(subs_labels)
    subs_labels[np.where(np.array(anomalies_annotation)==-1)] = anomaly_label

    return subs_labels.tolist()


def generate_neural_network_datasets(ts, subsequences_labels, m, test_normal_set_size=0.2) -> (list, list, list, list):
    """ Return train and test sets and indexes of subsequences which include in datasets
        
        Create datasets (train and test) for neural network
        """
    
    labels = np.unique(subsequences_labels)
    labels.sort()
    labels_dict = {}
    
    for label in labels:
        labels_dict[str(label)] = np.where(np.array(subsequences_labels) == label)[0].tolist()
    
    X_normal = []
    y_normal = []
    for i in range(labels[-1] + 1):
        for j in labels_dict[str(i)]:
            X_normal.append([j]+list(ts[j:j+m])) # idx, subsequence values
            y_normal.append(i)

    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(X_normal, y_normal, test_size=test_normal_set_size)

    train_set = [X_normal_train[i] + [y_normal_train[i]] for i in range(len(X_normal_train))] # idx, subsequence values, label
    test_set = [X_normal_test[i] + [y_normal_test[i]] for i in range(len(X_normal_test))] # idx, subsequence values, label

    anomaly_set = []
    for i in labels_dict[str(labels[0])]:
        anomaly_set.append([i]+list(ts[i:i+m])+[labels[0]]) # idx, subsequence values, label
    
    test_set = test_set + anomaly_set
    random.shuffle(test_set)
    
    train_set_idx = [item[0] for item in train_set] # idx
    test_set_idx = [item[0] for item in test_set] # idx
    
    train_set = [item[1:] for item in train_set] # subsequence values, label
    test_set = [item[1:] for item in test_set] # subsequence values, label
    
    return train_set, test_set, train_set_idx, test_set_idx


def create_statistics_tables(train_idx, test_idx, EDdist, MPdist) -> (list, list):
    """ Return statistic tables for training and test sets
        
        Create statistic tables for training and test sets which contains index and distance measurements (ED and MP distances)
    """
    
    train_statistics = []
    test_statistics = []
    
    for idx in train_idx:
        train_statistics.append([idx, EDdist[idx], MPdist[idx]])
    
    for idx in test_idx:
        test_statistics.append([idx, EDdist[idx], MPdist[idx]])
    
    return train_statistics, test_statistics


def create_snippets_set(snippets, snippets_num) -> list:
    """ Return snippets set
        
        Create snippets set
    """
    snippets_set = []
    for i in range(snippets_num):
        snippets_set.append(snippets[i]+[i])
    
    return snippets_set


def get_dataset_name(input_files, n, m, l, snippets_num):
    
    dataset_name = ''
    num_files = len(input_files)
    
    for i in range(num_files):
        dataset_name = dataset_name + input_files[i].split('.')[0] + '_'

    dataset_name = dataset_name + str(n) + '_' + str(m) + '_' + str(l) + '_' + str(snippets_num)

    return dataset_name


def write_dataset(set, dir, file_name):
    """
        Write dataset into csv file
    """
    
    with open(os.path.join(dir, file_name), 'w') as outfile:
        write = csv.writer(outfile)
        write.writerows(set)


def save_input_params(args, ts_lengths, n, dir, file_name):
    """
        Write input parameters into json file
    """

    input_params = {
        'input_files': args.input_files,
        'original_ts_lengths': ts_lengths,
        'lengths of input time series for train set': args.train_lengths,
        'n': n,
        'm': args.m,
        'l': args.l,
        'discords_fraction': args.alpha,
        'snippets_number': args.snippets_num,
        'test_normal_set_size': args.test_normal_set_size
    }

    with open(os.path.join(dir, file_name), 'w') as outfile:
        json.dump(input_params, outfile, indent=4)


def write_snippets(ts_snippets, dir, file_name):

    snippets_params = {
        'indices': ts_snippets['indices'].tolist(),
        'regimes': ts_snippets['regimes'].tolist()
    }
    
    print(snippets_params)
    print(snippets_params['indices'])
        
    with open(os.path.join(dir, file_name), 'w') as outfile:
        json.dump(snippets_params, outfile, indent=4)


def write_discords(ts_discords, dir, file_name):
    
    discords_params = {
        'indices': ts_discords['discords'].tolist(),
    }
    
    print(discords_params)
    print(discords_params['indices'])

    with open(os.path.join(dir, file_name), 'w') as outfile:
        json.dump(discords_params, outfile, indent=4)
