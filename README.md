# DiSSiD: Discord, Snippet, and Siamese Neural Networket-based Detector of anomalies
This repository is related to a semi-supervised method DiSSiD (Discord, Snippet, and Siamese Neural Networket-based Detector of anomalies) that detects subsequence anomalies in time series. DiSSiD is authored by Yana Kraeva (kraevaya@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the DiSSiD's source code (in Python), accompanying datasets, and experimental results. Please cite an article that describes DiSSiD as shown below.

The method is based on the concepts of discord and snippet, which formalize, respectively, the concepts of anomalous and typical time series subsequences. The proposed method includes a neural network model that calculates the anomaly score of the input subsequence and an algorithm to automatically construct the model’s training set. The model is implemented as a Siamese neural network, where we employ a modification of ResNet as a subnet. To train the model, we proposed a modified contrast loss function. The training set is formed as a representative fragment of the time series from which discords, low-fraction snippets with their nearest neighbors, and outliers within each snippet are removed since they are interpreted as abnormal, a typical activity of the subject, and noise, respectively.

# Citation
```
@article{Kraeva2024,
 author    = {Yana A. Kraeva},
 title     = {Detection of Time Series Anomalies Based on Data Mining and Neural Network Technologies},
 journal   = {Bulletin of the South Ural State University. Series: Computational Mathematics and Software Engineering},
 volume    = {12},
 number    = {3},
 pages     = {50-71},
 year      = {2023},
 doi       = {10.14529/cmse230304},
 url       = {https://doi.org/10.14529/cmse230304}
}
```
# Acknowledgement
This work was financially supported by the Russian Science Foundation (grant no. 23-21-00465).
