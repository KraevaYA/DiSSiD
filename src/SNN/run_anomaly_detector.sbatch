#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_outputs/slurm-%j.out


# ECG
##srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG805_snippet_10K_12000_250_75_2 -nn_type ResNet -act fit -epochs 30 -batch_size 64 -margin 1
##srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG805_snippet_10K_12000_250_75_2 -nn_type ResNet -act detect -epochs 30 -batch_size 64 -margin 1

srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG806_snippet_100K_120000_250_75_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG806_snippet_100K_120000_250_75_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1

srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG805_snippet_100K_120000_250_75_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
srun python anomaly_detector.py -dataset ECG/MBA_ECG803_ECG805_snippet_100K_120000_250_75_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1

##srun python anomaly_detector.py -dataset ECG/MBA_ECG803_805_806_snippet_100K_180000_250_75_3 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset ECG/MBA_ECG803_805_806_snippet_100K_180000_250_75_3 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1


#SMD
srun python anomaly_detector.py -dataset SMD/machine-1-5_14226_100_20_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
srun python anomaly_detector.py -dataset SMD/machine-1-5_14226_100_20_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1


#Daphnet
srun python anomaly_detector.py -dataset Daphnet/S09R01E0@7_S09R01E0@8_11520_216_72_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
srun python anomaly_detector.py -dataset Daphnet/S09R01E0@7_S09R01E0@8_11520_216_72_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1

##srun python anomaly_detector.py -dataset Daphnet/S01R02E0_17280_300_50_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset Daphnet/S01R02E0_17280_300_50_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1


#MITDB
srun python anomaly_detector.py -dataset MITDB/105@1-2_120000_520_75_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
srun python anomaly_detector.py -dataset MITDB/105@1-2_120000_520_75_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1

##srun python anomaly_detector.py -dataset MITDB/219@1-2_180000_546_273_2 -nn_type ResNet -act fit -epochs 50 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset MITDB/219@1-2_180000_546_273_2 -nn_type ResNet -act detect -epochs 50 -batch_size 128 -margin 1

##srun python anomaly_detector.py -dataset MITDB/118@1-2_120000_604_302_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset MITDB/118@1-2_120000_604_302_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1


# YAHOO
##srun python anomaly_detector.py -dataset YAHOO/Yahoo_A1real_20_data_853_60_30_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1 
##srun python anomaly_detector.py -dataset YAHOO/Yahoo_A1real_20_data_853_60_30_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1


# IOPS
##srun python anomaly_detector.py -dataset IOPS/KPI-49360_77406_1000_500_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset IOPS/KPI-49360_77406_1000_500_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1


# OPP
srun python anomaly_detector.py -dataset OPP/S3-ADL5_15722_400_100_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
srun python anomaly_detector.py -dataset OPP/S3-ADL5_15722_400_100_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1

##srun python anomaly_detector.py -dataset OPP/S4-ADL5_18149_300_50_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset OPP/S4-ADL5_18149_300_50_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1

##srun python anomaly_detector.py -dataset OPP/S3-ADL1_21781_300_60_2 -nn_type ResNet -act fit -epochs 30 -batch_size 128 -margin 1
##srun python anomaly_detector.py -dataset OPP/S3-ADL1_21781_300_60_2 -nn_type ResNet -act detect -epochs 30 -batch_size 128 -margin 1