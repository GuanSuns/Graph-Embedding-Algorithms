import sys
import csv
import os
import argparse
from evaluation import evaluation_utils

from gem.evaluation.evaluate_node_classification import TopKRanker

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def run_classification_experiment(emb_file, label_file, t_size=0.2, sampling=''):
    seed = 0
    print('Start running multi-label classification experiment ...')
    print('Embedding file: ', emb_file.split('/')[-1])
    print('Label file: ', label_file.split('/')[-1])

    # Pre-processing of emb_file and label_file
    patterns_sorted = evaluation_utils.emb_file_to_df(emb_file)
    encoded_sorted = evaluation_utils.k_encode_label_file(label_file)

    # Creating train test split 80% 20%
    X_train, X_test, y_train, y_test = train_test_split(patterns_sorted.values.T, encoded_sorted.values.T, test_size=t_size, random_state=seed)

    # classifier list
    clf_list = [
        # DecisionTreeClassifier(random_state=seed),
        # KNeighborsClassifier(n_neighbors=3),
        # MLPClassifier(random_state=seed, max_iter=500),
        # RandomForestClassifier(random_state=seed),
        # TopKRanker(LogisticRegression(solver='liblinear')), # OneVsRest
        # OneVsRestClassifier(LogisticRegression(solver='lbfgs')), # Terrible OneVsRest
        TopKRanker(LogisticRegression(solver='lbfgs'))  # OneVsRest
    ]

    result = []
    for clf in clf_list:
        print(clf.__class__.__name__)
        (macro_f1, micro_f1) = evaluate_clf(clf, X_train, X_test, y_train, y_test)
        record = {'emb_file': emb_file.split('/')[-1], 'classifier': clf.__class__.__name__, 'macro_f1': macro_f1, 'micro_f1': micro_f1, 'test_set_size': t_size, 'sampling': sampling}
        result.append(record)
        print('\n')
    return result


def evaluate_clf(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    prediction = None
    if type(clf) is TopKRanker:  # OneVsRest
        top_k_list = list(y_test.sum(axis=1))
        prediction = clf.predict(X_test, top_k_list)
    else:
        prediction = clf.predict(X_test)
    if prediction is None:
        print("run_classification_experiment error - prediction was not run.")
        return

    macro_f1 = f1_score(y_test, prediction, average='macro')
    micro_f1 = f1_score(y_test, prediction, average='micro')
    print("Macro f1: ", macro_f1)
    print("Micro f1: ", micro_f1)
    return macro_f1, micro_f1


def run_evaluation():
    emb_files = [
        'ppi_Node2Vec-Embedding_1574398302.065598.emb',
        'ppi_Node2Vec-Embedding_1574398447.8591008.emb',
        'ppi_Node2Vec-Embedding_1574398527.519395.emb',
        'ppi_Node2Vec-Embedding_1574398649.3481941.emb',
        'ppi_Node2Vec-Embedding_1574398765.699774.emb',
    ]

    sampling_files = [
        'biased-random-walk-dfs-1574398300.780365',
        'node2vec-random-walk-1574398446.602928',
        'simple_random_walk-1574398526.254327',
        'approximate-bfs-walk-1574398648.08918',
        'approximate-dfs-walk-1574398764.39482',
    ]
    emb_file_dir = './output/ppi/'
    label_file = './data/ppi/ppi-labels.txt'

    eval_result_fname = 'evaluation_results.csv'
    field_names = ['emb_file', 'sampling',  'classifier', 'macro_f1', 'micro_f1', 'test_set_size']
    if not os.path.exists(eval_result_fname):
        # ['emb_file', 'classifier', 'macro_f1', 'micro_f1', 'test_set_size']
        csv_file = csv.DictWriter(open(eval_result_fname, 'w'), fieldnames=field_names)
        csv_file.writeheader()
    else:
        csv_file = csv.DictWriter(open(eval_result_fname, 'a'), fieldnames=field_names)

    for i, emb_file in enumerate(emb_files):
        for test_size_portion in range(1, 10):
            result = run_classification_experiment(emb_file_dir + emb_file, label_file, test_size_portion / 10.0, sampling=sampling_files[i])
            for result_item in result:
                csv_file.writerow(result_item)


if __name__ == '__main__':
    run_evaluation()
