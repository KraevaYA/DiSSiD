import numpy as np
import pandas as pd
import math
import os
import csv
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import siamese_nn
import utils
import config
import plots
import accuracy


def main():
    
    args = utils.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        print(e)
    
    dataset_dir = os.path.join(config.SNN_DATASETS_DIR, args.dataset)
    results_dir = os.path.join(config.RESULTS_DIR, args.dataset)
    utils.create_directory(results_dir)

    dataset_params = utils.read_json_file(os.path.join(dataset_dir, 'input_params.json'))
    
    if (args.act == 'fit'):

        # 1. read and load the training set
        print('1. Start to read and transform the training set for fitting the SNN')
        train_val_set_path = os.path.join(dataset_dir, 'train_set.csv')
        x_train_val, y_train_val = utils.load_dataset(train_val_set_path)
        x_train_val = utils.normalize_dataset(x_train_val)
        x_train, y_train, x_val, y_val = utils.train_val_split(x_train_val, y_train_val, args.val_size)
        
        snippets_set_path = os.path.join(dataset_dir, 'snippets.csv')
        x_snippets, y_snippets = utils.load_dataset(snippets_set_path)
        x_snippets = utils.normalize_dataset(x_snippets)
        
        # make the train pairs
        pairs_train, pairs_labels_train = utils.make_train_set_pairs(x_train, y_train)
        # make the validation pairs
        pairs_val, pairs_labels_val = utils.make_train_set_pairs(x_val, y_val)
    
        # split the training pairs
        x_train_1 = pairs_train[:, 0]
        x_train_2 = pairs_train[:, 1]

        # split the validation pairs
        x_val_1 = pairs_val[:, 0]
        x_val_2 = pairs_val[:, 1]

        print('The training and validation sets are ready to fit the SNN\n')
        
        # -----------------------------------------------

        # 2. init SNN, fit and predict the similarity score
        print('2. Start to fit SNN')
        SNN_results_dir = os.path.join(results_dir, args.nn_type)
        utils.create_directory(SNN_results_dir)
        
        input_shape = (x_train_1.shape[1], 1)
        SNN_model = siamese_nn.SiameseNN(SNN_results_dir, input_shape, args.nn_type, args.epochs, args.batch_size, args.margin, args.l_emb, args.mpdist_k, save_params=True)
            
        if len(x_train_1.shape) == 2:  # if univariate
            x_train_1 = x_train_1.reshape((x_train_1.shape[0], x_train_1.shape[1], 1))
            x_train_2 = x_train_2.reshape((x_train_2.shape[0], x_train_2.shape[1], 1))
            x_val_1 = x_val_1.reshape((x_val_1.shape[0], x_val.shape[1], 1))
            x_val_2 = x_val_2.reshape((x_val_2.shape[0], x_val_2.shape[1], 1))

        SNN_model.fit(x_train_1, x_train_2, x_val_1, x_val_2, pairs_labels_train, pairs_labels_val)
        print('SNN is fitted\n')

        train_time_path = os.path.join(SNN_results_dir, 'train_time.csv')
        with open(train_time_path, 'w', newline='') as file:
            write = csv.writer(file, delimiter='\n')
            write.writerow((SNN_model.train_time,))

        # --------------------------------------------------------------

        # 3. read and load the test and snippets sets
        print("3. Start to read and transform the test set for finding the anomaly threshold")
        test_set_path = os.path.join(dataset_dir, 'test_set.csv')
        x_test, y_test = utils.load_dataset(test_set_path)
        x_test = utils.normalize_dataset(x_test)

        pairs_test, pairs_labels_test = utils.make_test_set_pairs(x_test, y_test, x_snippets, y_snippets)

        # split the test pairs
        x_test_1 = pairs_test[:, 0]
        x_test_2 = pairs_test[:, 1]

        x_test_1 = x_test_1.reshape((x_test_1.shape[0], x_test_1.shape[1], 1))
        x_test_2 = x_test_2.reshape((x_test_2.shape[0], x_test_2.shape[1], 1))

        similarity_scores, test_time = SNN_model.predict(x_test_1, x_test_2)
        print('The Siamese neural network finished to detect anomalies in the test set\n')

        # --------------------------------------------------------------

        # 4. find anomaly threshold
        print('4. Start to find anomaly threshold using the SNN results on the test set')
        results_table = utils.make_results_table(similarity_scores, pairs_labels_test, y_test, dataset_params['snippets_number'])

        plots_dir = os.path.join(results_dir, args.nn_type+'/plots')
        utils.create_directory(plots_dir)
        plots.plot_score_probability(results_table, os.path.join(plots_dir, 'score_probability.png'))

        normal_similarity_scores = np.sort(results_table[results_table['real_label']=='0.0']['min_similarity_score'].values)
        anomaly_threshold = utils.calculate_threshold(normal_similarity_scores, config.N_PERCENTILE)
        print('The anomaly threshold is calculated\n')

        utils.save_snn_params(args.nn_type, args.epochs, args.batch_size, args.margin, args.optimizer, anomaly_threshold, os.path.join(SNN_results_dir, 'snn_params.json'))

        print('Fitting the SNN and the finding the anomaly threshold are done\n')

    else:
        
        snn_params = utils.read_json_file(os.path.join(results_dir, args.nn_type+'/snn_params.json'))
        
        test_original_ts_path = os.path.join(dataset_dir, 'test_original_ts.csv')
        test_ts = utils.load_test_original_ts_from_csv(test_original_ts_path)
        
        test_label_path = os.path.join(dataset_dir, 'test_label.csv')
        true_label = utils.load_test_original_ts_from_csv(test_label_path)

        N = len(test_ts) - dataset_params['m'] + 1
        x_test = utils.split_ts_to_subs(test_ts, N, dataset_params['m'])
        x_test = utils.normalize_dataset(x_test)
    
        snippets_set_path = os.path.join(dataset_dir, 'snippets.csv')
        x_snippets, y_snippets = utils.load_dataset(snippets_set_path)
        x_snippets = utils.normalize_dataset(x_snippets)

        pairs_test = utils.make_original_test_ts_pairs(x_test, x_snippets)
    
        # split the test pairs
        x_test_1 = pairs_test[:, 0]
        x_test_2 = pairs_test[:, 1]

        x_test_1 = x_test_1.reshape((x_test_1.shape[0], x_test_1.shape[1], 1))
        x_test_2 = x_test_2.reshape((x_test_2.shape[0], x_test_2.shape[1], 1))

        SNN_results_dir = os.path.join(results_dir, args.nn_type)
    
        input_shape = (x_test_1.shape[1], 1)
        SNN_model = siamese_nn.SiameseNN(SNN_results_dir, input_shape, args.nn_type, snn_params['epochs'], snn_params['batch_size'], snn_params['margin'], save_params=False)
        pairs_similarity_scores, test_time = SNN_model.predict(x_test_1, x_test_2)
        
        test_time_path = os.path.join(SNN_results_dir, 'test_time.csv')
        with open(test_time_path, 'w', newline='') as file:
            write = csv.writer(file, delimiter='\n')
            write.writerow((test_time,))
        
        plots_dir = os.path.join(results_dir, args.nn_type+'/plots')
        utils.create_directory(plots_dir)
        predicted_scores = utils.find_subs_similarity_scores(pairs_similarity_scores, dataset_params['snippets_number'])
        predicted_scores = np.array(predicted_scores)
        predicted_scores[predicted_scores>1] = 1
        predicted_scores = predicted_scores.tolist()
        plots.plot_similarity_scores(test_ts.reshape(1,-1)[0], predicted_scores, true_label.reshape(1,-1)[0], N, os.path.join(plots_dir, 'similarity_scores.png'))
        
        # calculate accuracy matrics with VUS package
        if (config.ACCURACY_METHOD == 'VUS'):
        
            np_predicted_scores = pd.DataFrame(predicted_scores).to_numpy().reshape(1,-1)[0]
            np_true_label = pd.DataFrame(true_label).to_numpy().reshape(1,-1)[0]
            accuracy_metrics = accuracy.scoring(np_predicted_scores, np_true_label, slidingWindow=dataset_params['m'])
            utils.save_metrics(accuracy_metrics, os.path.join(SNN_results_dir, 'accuracy_metrics.csv'))
        
        else:
            if (config.ACCURACY_METHOD == 'OWN'):
                anomaly_regions = utils.find_anomaly_regions(subs_similarity_scores, snn_params['anomaly_threshold'])
        
                real_anomaly_annotation = utils.get_real_anomaly_annotation(dataset_params['input_files'], dataset_params['original_ts_lengths'], dataset_params['lengths of input time series for train set']) # get real anomaly index

                predicted_anomalies_idx = utils.find_top_k_predicted_anomalies(subs_similarity_scores, len(real_anomaly_annotation), dataset_params['m'])
                precision = utils.calculate_precision(real_anomaly_annotation, predicted_anomalies_idx, dataset_params['m'])
        

if __name__ == '__main__':
    main()
