import sys
import csv
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


def run_classification_experiment(emb_file, label_file, t_size=0.2):
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
        DecisionTreeClassifier(random_state=seed),
        KNeighborsClassifier(n_neighbors=3),
        MLPClassifier(random_state=seed, max_iter=500),
        RandomForestClassifier(random_state=seed),
        # TopKRanker(LogisticRegression(solver='liblinear')), # OneVsRest
        # OneVsRestClassifier(LogisticRegression(solver='lbfgs')), # Terrible OneVsRest
        TopKRanker(LogisticRegression(solver='lbfgs'))  # OneVsRest
    ]

    result = []
    for clf in clf_list:
        print(clf.__class__.__name__, "\n")
        (macro_f1, micro_f1) = evaluate_clf(clf, X_train, X_test, y_train, y_test)
        result.append((emb_file.split('/')[-1], clf.__class__.__name__, t_size, macro_f1, micro_f1))
        print("\n\n")
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
        'blog-catalog_CBOW-Embedding_1574113020.0535862.emb',
        'blog-catalog_FastText-Embedding_1574111311.4566038.emb',
        'blog-catalog_GloVe-Embedding_1574115685.46.emb',
        'blog-catalog_Node2Vec-Embedding_1574111286.549976.emb'
    ]

    emb_file_dir = './output/blog-catalog/'
    label_file = './data/blog-catalog-deepwalk/blog-catalog-labels.txt'

    csv_file = csv.writer(open('evaluation_results.csv', 'a'))

    for emb_file in emb_files:
        for i in range(1, 10):
            result = run_classification_experiment(emb_file_dir + emb_file, label_file, i / 10)
            for result_item in result:
                csv_file.writerow(result_item)


if __name__ == '__main__':
    run_evaluation()
