#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_outputs/slurm-%j.out


# ECG
srun python anomaly_detector.py -dataset ECG/MBA_ECG803_805_806_snippet_100K_180000_250_75_3 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
srun python anomaly_detector.py -dataset ECG/MBA_ECG803_805_806_snippet_100K_180000_250_75_3 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9

srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG806_snippet_100K_120000_250_75_2 -nn_type ResNet -act fit -epochs 1 -batch_size 128 -margin 1 -l_emb 40 -mpdist_k 0.6
srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG806_snippet_100K_120000_250_75_2 -nn_type ResNet -act detect -epochs 1 -batch_size 128 -margin 1 -l_emb 40 -mpdist_k 0.6

#srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG805_snippet_100K_120000_250_75_2 -nn_type ResNet -act fit -epochs 1 -batch_size 128 -margin 1 -l_emb 40 -mpdist_k 0.6
#srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG805_snippet_100K_120000_250_75_2 -nn_type ResNet -act detect -epochs 1 -batch_size 128 -margin 1 -l_emb 40 -mpdist_k 0.6

srun python anomaly_detector.py -dataset ECG/MBA_ECG805_ECG806_snippet_100K_120000_250_75_2 -nn_type ResNet -act fit -epochs 1 -batch_size 128 -margin 1 -l_emb 40 -mpdist_k 0.6
srun python anomaly_detector.py -dataset ECG/MBA_ECG805_ECG806_snippet_100K_120000_250_75_2 -nn_type ResNet -act detect -epochs 1 -batch_size 128 -margin 1 -l_emb 40 -mpdist_k 0.6


# Daphnet
srun python anomaly_detector.py -dataset Daphnet/S01R02E0_17280_300_50_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
srun python anomaly_detector.py -dataset Daphnet/S01R02E0_17280_300_50_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9

srun python anomaly_detector.py -dataset Daphnet/S09R01E0@7_S09R01E0@8_11520_216_72_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 48 -mpdist_k 0.9
srun python anomaly_detector.py -dataset Daphnet/S09R01E0@7_S09R01E0@8_11520_216_72_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 48 -mpdist_k 0.9


# MITDB
srun python anomaly_detector.py -dataset MITDB/105@1-2_120000_520_75_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 16 -mpdist_k 0.9
srun python anomaly_detector.py -dataset MITDB/105@1-2_120000_520_75_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 16 -mpdist_k 0.9

##srun python anomaly_detector.py -dataset MITDB/118@1-2_120000_604_302_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.05
##srun python anomaly_detector.py -dataset MITDB/118@1-2_120000_604_302_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.05

##srun python anomaly_detector.py -dataset MITDB/219@1-2_180000_546_273_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.05
##srun python anomaly_detector.py -dataset MITDB/219@1-2_180000_546_273_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.05

##srun python anomaly_detector.py -dataset MITDB/105_118@1_120000_453_151_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 48 -mpdist_k 0.5
##srun python anomaly_detector.py -dataset MITDB/105_118@1_120000_453_151_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 48 -mpdist_k 0.5


# IOPS
##srun python anomaly_detector.py -dataset IOPS/KPI-49360_77406_1000_500_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
##srun python anomaly_detector.py -dataset IOPS/KPI-49360_77406_1000_500_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9


# SMD
srun python anomaly_detector.py -dataset SMD/machine-1-5_14226_100_20_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 32 -mpdist_k 0.9
srun python anomaly_detector.py -dataset SMD/machine-1-5_14226_100_20_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 32 -mpdist_k 0.9


# OPP
##srun python anomaly_detector.py -dataset OPP/S3-ADL5_15722_400_100_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
##srun python anomaly_detector.py -dataset OPP/S3-ADL5_15722_400_100_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9

##srun python anomaly_detector.py -dataset OPP/S4-ADL5_18149_300_50_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
##srun python anomaly_detector.py -dataset OPP/S4-ADL5_18149_300_50_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9


# YAHOO
srun python anomaly_detector.py -dataset YAHOO/Yahoo_A1real_20_data_853_60_30_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
srun python anomaly_detector.py -dataset YAHOO/Yahoo_A1real_20_data_853_60_30_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1 -l_emb 64 -mpdist_k 0.9
