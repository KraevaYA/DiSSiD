# coding: utf-8

import numpy as np
import os

import config
import matrix_profile
import top_k_discords
import snippets
import snippets_anomalies
import plots
import utils


def main():

    args = utils.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        print(e)

    original_train_ts, original_test_ts, train_label, test_label, ts_lengths = utils.split_ts(args.input_files, args.train_lengths)

    n = len(original_train_ts)
    N = n - args.m + 1

    dataset_name = utils.get_dataset_name(args.input_files, n, args.m, args.l, args.snippets_num)
    snn_dataset_dir = os.path.join(config.SNN_DATASETS_DIR, dataset_name)
    utils.create_directory(snn_dataset_dir)

    plots_dir = os.path.join(config.PLOTS_DIR, dataset_name)
    utils.create_directory(plots_dir)
    image_plots_dir = os.path.join(plots_dir, 'images')
    utils.create_directory(image_plots_dir)
    data_plots_dir = os.path.join(plots_dir, 'data')
    utils.create_directory(data_plots_dir)

    # plot time series
    plots.plot_ts(original_train_ts, n, os.path.join(image_plots_dir, 'train_ts.png'), title='Train Time Series')
    utils.write_dataset(np.array(original_train_ts).reshape(-1,1), data_plots_dir, 'train_ts.csv')
    utils.write_dataset(np.array(train_label).reshape(-1,1), data_plots_dir, 'train_label.csv')

    # 1. find matrix profile
    print("1. Start to find Matrix Profile")
    mp = matrix_profile.find_mp(original_train_ts, args.m)
    plots.plot_ts(mp['mp'], len(mp['mp']), os.path.join(image_plots_dir, 'matrix_profile.png'), title='Matrix Profile')
    utils.write_dataset(np.array(mp['mp']).reshape(-1,1), data_plots_dir, 'matrix_profile.csv')
    print("Matrix Profile is computed\n")
    
    # 2. find discords in time series
    print("2. Start to find discords in the time series")
    discords_num = int(np.ceil(args.alpha*N))
    ts_discords = top_k_discords.find_discords(mp, args.m, discords_num)
    discords_idx = list(ts_discords['discords'])
    discords_idx.sort()
    discords_annotation = top_k_discords.construct_discords_annotation(discords_idx, N, args.m)
    plots.plot_discords(original_train_ts, mp['mp'], ts_discords['discords'], n, args.m, N, discords_num, os.path.join(image_plots_dir, 'discords.png'))
    utils.write_discords(ts_discords, data_plots_dir, 'discords.json')
    utils.write_dataset(np.array(discords_annotation).reshape(-1,1), data_plots_dir, 'discords_annotation.csv')
    print(f"{discords_num} Discords are founded in the time series\n")
    
    # 3. find snippets in the time series
    print("3. Start to find snippets in the time series")
    if (config.SNIPPET_FIND_WITH_OPTIMIZATION):
        ts_snippets = snippets.find_snippets_with_optimization(original_train_ts, args.m, args.l, args.snippets_num)
    else:
        ts_snippets = snippets.find_snippets_without_optimization(np.array(original_train_ts), args.m, args.l, args.snippets_num)
    
    profiles_curve = snippets.find_profiles_curve(ts_snippets['profiles'], args.snippets_num)

    plots.plot_snippets(original_train_ts, ts_snippets, n, args.m, args.snippets_num, os.path.join(image_plots_dir, 'snippets.png'))
    utils.write_snippets(ts_snippets, data_plots_dir, 'snippets.json')
    plots.plot_profiles(ts_snippets['profiles'], len(ts_snippets['profiles'][0]), args.snippets_num, os.path.join(image_plots_dir, 'mpdist_profiles.png'))
    utils.write_dataset([profile.tolist() for profile in ts_snippets['profiles'].T], data_plots_dir, 'mpdist_profiles.csv')
    print("3. Snippets are founded in the time series\n")
    
    # 4. find the snippets anomalies
    print("4. Start to find snippets anomalies in the time series")
    moving_max_profiles = []
    w = args.m+1 # window width
    
    for i in range(args.snippets_num):
        moving_max_profiles.append(utils.moving_max(ts_snippets['profiles'][i], w))
    
    if (config.SNIPPETS_ANOMALY_METHOD == 'IsolationForest'):
        print(config.SNIPPETS_ANOMALY_METHOD)
        max_mpdist_regimes = snippets.find_mpdist_regimes(ts_snippets['regimes'], moving_max_profiles, args.snippets_num)
        max_regimes_profiles = snippets.find_regimes_profiles(max_mpdist_regimes, N, args.snippets_num)
        #ts_snippets_anomalies = snippets_anomalies.find_snippets_anomalies_IF(max_regimes_profiles, args.snippets_num)
        ts_snippets_anomalies = snippets_anomalies.find_snippets_anomalies_IF(moving_max_profiles, args.snippets_num)
    else:
    # if KNN
        print(config.SNIPPETS_ANOMALY_METHOD)
        max_mpdist_regimes = snippets.find_mpdist_regimes(ts_snippets['regimes'], moving_max_profiles, args.snippets_num)
        max_mpdist_all_regimes = snippets.find_mpdist_all_regimes(ts_snippets['regimes'], moving_max_profiles, args.snippets_num)
        ts_snippets_anomalies = snippets_anomalies.find_snippets_anomalies_KNN(max_mpdist_all_regimes, ts_snippets['indices'], N, args.snippets_num)

    print(len(ts_snippets_anomalies))
    
    snippets_anomalies_annotation = snippets_anomalies.construct_snippets_anomalies_annotation(max_mpdist_regimes, ts_snippets_anomalies, ts_snippets['indices'], N+1, args.snippets_num, args.m)
    plots.plot_annotation(original_train_ts, train_label, snippets_anomalies_annotation, len(snippets_anomalies_annotation), os.path.join(image_plots_dir, 'snippets_anomalies_annotation.png'), title="Snippets anomalies annotation")
    utils.write_dataset(np.array(snippets_anomalies_annotation).reshape(-1,1), data_plots_dir, 'snippets_anomalies_annotation.csv')
    print("4. Snippets anomalies are founded in the time series\n")

    # 5. find the anomaly annotation
    print("5. Start to find anomaly annotation")
    print(len(discords_annotation))
    print(len(snippets_anomalies_annotation))
    anomalies_annotation = snippets_anomalies.construct_anomalies_annotation(discords_annotation, snippets_anomalies_annotation[:len(discords_annotation)])
    plots.plot_annotation(original_train_ts, train_label, anomalies_annotation, len(anomalies_annotation), os.path.join(image_plots_dir, 'anomalies_annotation.png'), title="Anomalies annotation")
    utils.write_dataset(np.array(anomalies_annotation).reshape(-1,1), data_plots_dir, 'anomalies_annotation.csv')
    print("5. Anomaly annotation is founded\n")

    # 6. generate the train and test sets for Siamese Neural Network
    print("6. Start to generate the train and test sets for Siamese Neural Network")
    subsequences_labels = utils.label_subsequences(ts_snippets, anomalies_annotation, len(anomalies_annotation), args.m, args.snippets_num) #ts_snippets['fractions'].tolist())
    utils.write_dataset(np.array(subsequences_labels).reshape(-1,1), data_plots_dir, 'subsequences_labels.csv')

    sets = []
    train_set, test_set, train_set_idx, test_set_idx = utils.generate_neural_network_datasets(original_train_ts, subsequences_labels, args.m, args.test_normal_set_size)
    sets.append(train_set)
    sets.append(test_set)
    print("6. Generation of the train and test sets for Siamese Neural Network finished\n")

    # 7. create the snippet set
    print("7. Start to create the snippet set")
    snippets_set = utils.create_snippets_set(ts_snippets['snippets'].tolist(), args.snippets_num)
    sets.append(snippets_set)
    print("7. The snippet set is created\n")

    # 8. create the statistics tables for train and test sets
    print("8. Start to create the statistics tables")
    train_statistics, test_statistics = utils.create_statistics_tables(train_set_idx, test_set_idx, mp['mp'], profiles_curve)
    sets.append(train_statistics)
    sets.append(test_statistics)
    print("8. Statistics tables are created\n")

    # 9. write datasets
    print("9. Start to write datasets for SNN")
    for i in range(len(sets)):
        utils.write_dataset(sets[i], snn_dataset_dir, config.OUTFILE_NAMES[i])
    print("9. Datasets for SNN are written and saved\n")

    # 10. write the original test time series and label
    print("10. Start to write the original test time series and label for SNN")
    utils.write_dataset(np.array(original_test_ts).reshape(-1,1), snn_dataset_dir, config.OUTFILE_NAMES[-3])
    utils.write_dataset(np.array(test_label).reshape(-1,1), snn_dataset_dir, config.OUTFILE_NAMES[-2])
    print("10. The original test time series and label are written and saved\n")
    
    # 11. write the input parameters into the json file
    print("11. Start to write the input parameters into the json file")
    utils.save_input_params(args, ts_lengths, n, snn_dataset_dir, config.OUTFILE_NAMES[-1])
    print("11. The input parameters ate written and saved\n")


if __name__ == '__main__':
    main()
