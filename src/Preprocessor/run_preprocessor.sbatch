#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_outputs/slurm-%j.out

## ECG
##srun python preprocessor.py -files ECG/MBA_ECG803_data_snippet_100K.out ECG/MBA_ECG806_data_snippet_100K.out -train_lengths 60000 60000 -m 250 -l 75 -alpha 0.0003 -snippets_num 2
##srun python preprocessor.py -files ECG/MBA_ECG803_data_snippet_100K.out ECG/MBA_ECG805_data_snippet_100K.out -train_lengths 60000 60000 -m 250 -l 75 -alpha 0.0008 -snippets_num 2
##srun python preprocessor.py -files ECG/MBA_ECG805_data_snippet_100K.out ECG/MBA_ECG806_data_snippet_100K.out -train_lengths 60000 60000 -m 250 -l 75 -alpha 0.0007 -snippets_num 2
srun python preprocessor.py -files ECG/MBA_ECG803_data_snippet_100K.out ECG/MBA_ECG805_data_snippet_100K.out ECG/MBA_ECG806_data_snippet_100K.out -train_lengths 60000 60000 60000 -m 250 -l 75 -alpha 0.001 -snippets_num 3


## MITDB
##srun python preprocessor.py -files MITDB/105.test.csv@1_snippet_100K.out MITDB/105.test.csv@2_snippet_100K.out -train_lengths 60000 60000 -m 520 -l 75 -alpha 0.0002 -snippets_num 2
##srun python preprocessor.py -files MITDB/219.test.csv@1_snippet_100K.out MITDB/219.test.csv@2_snippet_100K.out -train_lengths 60000 60000 -m 546 -l 273 -alpha 0.0001 -snippets_num 2
##srun python preprocessor.py -files MITDB/118.test.csv@1_snippet_100K.out MITDB/118.test.csv@2_snippet_100K.out -train_lengths 60000 60000 -m 604 -l 302 -alpha 0.0001 -snippets_num 2


## SMD
##srun python preprocessor.py -files SMD/machine-1-5.test.csv@19.out -train_lengths 14226 -m 100 -l 20 -alpha 0.0005 -snippets_num 2
##srun python preprocessor.py -files SMD/machine-1-6.test.csv@9.out -train_lengths 14213 -m 120 -l 20 -alpha 0.005 -snippets_num 2
##srun python preprocessor.py -files SMD/machine-1-1.test.csv@12.out -train_lengths 17087 -m 70 -l 20 -alpha 0.0005 -snippets_num 2


## OPP
##srun python preprocessor.py -files OPP/S3-ADL5.test.csv@38.out -train_lengths 15722 -m 200 -l 50 -alpha 0.0005 -snippets_num 2
##srun python preprocessor.py -files OPP/S4-ADL5.test.csv@65.out -train_lengths 18149 -m 300 -l 50 -alpha 0.0005 -snippets_num 2
##srun python preprocessor.py -files OPP/S3-ADL1.test.csv@50.out -train_lengths 21781 -m 300 -l 60 -alpha 0.0005 -snippets_num 2


## Daphnet
##srun python preprocessor.py -files Daphnet/S01R02E0.test.csv@4.out -train_lengths 17280 -m 300 -l 50 -alpha 0.0001 -snippets_num 2
##srun python preprocessor.py -files Daphnet/S09R01E0.test.csv@7.out Daphnet/S09R01E0.test.csv@8.out -train_lengths 5760 5760 -m 216 -l 72 -alpha 0.0005 -snippets_num 2


##Yahoo
##srun python preprocessor.py -files YAHOO/Yahoo_A1real_20_data.out -train_lengths 853 -m 60 -l 30 -alpha 0.001 -snippets_num 2
##srun python preprocessor.py -files YAHOO/Yahoo_A1real_19_data.out -train_lengths 854 -m 50 -l 20 -alpha 0.005 -snippets_num 2


##IOPS
##srun python preprocessor.py -files IOPS/KPI-49360.train.out -train_lengths 77406 -m 1000 -l 500 -alpha 0.0001 -snippets_num 2

