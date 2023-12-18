# coding: utf-8

# Directories and files
ORIGINAL_TS_DIR = '../../datasets/original_ts' # directory with original real time series
SNN_DATASETS_DIR = '../../datasets/SNN_datasets' # directory with training and test sets for SNN
OUTFILE_NAMES = ['train_set.csv', 'test_set.csv', 'snippets.csv', 'train_statistics.csv', 'test_statistics.csv', 'test_original_ts.csv', 'test_label.csv', 'input_params.json']
PLOTS_DIR = '../../plots/Preprocessor'

SNIPPETS_ANOMALY_METHOD = 'KNN' #'KNN' or 'IsolationForest'
IF_CONTAMINATION = 0.05

# parameters for creating the training and test sets
ALPHA = 0.0008 # fraction of anomalies in time series

TOP_K_PROFILES_PER_SNIPPET = 30
SNIPPETS_FRACTION_THRESHOLD = 0.05
SNIPPET_FIND_WITH_OPTIMIZATION = False

TEST_NORMAL_SET_SIZE = 0.2 # fraction of normal samples for test set
