# coding: utf-8

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from kneed import KneeLocator

import config


def find_snippets_anomalies_IF(regimes_profiles, snippets_num):
    '''
        Find outliers, which are not similar to snippets using Isolation Forest algorithm.
        Isolation Forest returns -1 for outliers and 1 for inliers.
    
        Parameters
        ----------
        regimes_profiles :
        snippets_num : The number of snippets
        
    '''

    anomaly_label = -1
    data = np.array(regimes_profiles).T

    model = IsolationForest(random_state=0, contamination=config.IF_CONTAMINATION)
    model.fit(data)
    labels = model.predict(data)

    snippets_anomalies = np.where(labels==anomaly_label)[0]
    
    return snippets_anomalies


def find_snippets_anomalies_KNN(regimes_profiles, snippets_indices, N, snippets_num):
    """
        Find outliers, which are not similar to snippets using Isolation Forest algorithm.
        KNN returns -1 for outliers and 1 for inliers.
        
        Parameters
        ----------
        regimes_profiles :
        
        snippets_indices :
        
        N : int
        
        snippets_num : int
        The number of snippets.
        
        Returns
        ----------
        snippets_anomalies :
    """
    
    anomaly_label = -1
    normal_label = 1
    anomaly_snippets_profile = [normal_label]*N
    
    for i in range(snippets_num):
        
        SSE = []
        for num_clusters in range(1,10,1):
            km = KMeans(n_clusters=num_clusters, init='k-means++')
            km.fit(regimes_profiles[i])
            SSE.append(km.inertia_)
        
        # find optimal number of clusters
        indx = np.arange(len(SSE))
        knee = KneeLocator(indx, SSE, S=1, curve='convex', direction='decreasing', interp_method='polynomial')
        optimal_num_clusters = knee.elbow + 1
        print(optimal_num_clusters)
        
        # cluster with the optimal number of clusters
        km = KMeans(n_clusters=optimal_num_clusters, init='k-means++')
        km.fit(regimes_profiles[i])
        
        # define the cluster number where the snippet is located
        cluster_snippet = km.labels_[regimes_profiles[i].index.get_loc(snippets_indices[i])]
        
        cluster_snippet_center = km.cluster_centers_[cluster_snippet]

        max_dist = 0
        for j in range(optimal_num_clusters):
            if (j != cluster_snippet):
                cluster_center =  km.cluster_centers_[j]
                dist = np.linalg.norm(cluster_center - cluster_snippet_center)
                if (max_dist < dist):
                    max_dist = dist
                    abnormal_cluster = j
        
        for j in range(len(km.labels_)):
            if (km.labels_[j] == abnormal_cluster):
                global_ind = regimes_profiles[i].index[j]
                anomaly_snippets_profile[global_ind] = anomaly_label

    snippets_anomalies = np.where(np.array(anomaly_snippets_profile)==anomaly_label)[0]

    return snippets_anomalies


def construct_snippets_anomalies_annotation(regimes, snippets_anomalies, snippets_idxs, n, snippets_num, m):
    '''
        
        Parameters
        ----------
        regimes :
        snippets_anomalies :
        snippets_num : The number of snippets
        
    '''
    anomaly_label = -1
    normal_label = 1
    
    snippets_anomalies_annotation = [normal_label]*n
    
    for idx in snippets_anomalies:
        snippets_anomalies_annotation[idx] = anomaly_label
    
    snippets_anomalies_annotation = np.array(snippets_anomalies_annotation)
    
    for i in range(snippets_num):
        regime_labels = snippets_anomalies_annotation[regimes[i].index]
        normal_regime_labels_idx = np.where(regime_labels==normal_label)[0]
        min_normal_mpdist = regimes[i].iloc[normal_regime_labels_idx].min()
        
        anomaly_idx = np.where(regime_labels==anomaly_label)[0]
        anomaly_mpdist_vector = regimes[i].iloc[anomaly_idx]
        
        changed_idx = anomaly_mpdist_vector[anomaly_mpdist_vector < min_normal_mpdist].index
        snippets_anomalies_annotation[changed_idx] = normal_label
    
    for i in range(snippets_num):
        snippet_idx = snippets_idxs[i]
        snippets_anomalies_annotation[snippet_idx-m:snippet_idx+m] = 1
    
    return snippets_anomalies_annotation


def construct_anomalies_annotation(discords_annotation, snippets_anomalies_annotation):
    '''

        Parameters
        ----------
        discords_annotation :
        snippets_anomalies_annotation :
        
    '''
    
    anomalies_annotation = np.fmin(discords_annotation, snippets_anomalies_annotation).tolist()
    
    return anomalies_annotation


