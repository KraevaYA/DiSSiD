# coding: utf-8

# Directories and files
SNN_DATASETS_DIR = '../../datasets/SNN_datasets' # directory with training and test sets for SNN
RESULTS_DIR = '../../SNN_results' # directory where weights of classifier are saved
ANNOTATION_DIR = '../../datasets/original_ts' # directory with training and test sets for SNN


#parameters for neural network
NN_TYPE = 'ResNet' # 'FCN' or 'Inception' or 'MLP' or 'ResNet'
ACT = 'detect' # 'fit' the neural network or 'predict' items (subsequences) of the test dataset
EPOCHS = 50
BATCH_SIZE = 64
MARGIN = 1
OPTIMIZER = 'adam'
#M_EMB = 128
L_EMB = 64
MPDIST_K = 0.8

VAL_SIZE = 0.1
N_PERCENTILE = 100
ACCURACY_METHOD = 'VUS' # or 'own'

