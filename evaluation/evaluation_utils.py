import numpy as np
import pandas as pd


def k_encode_label_file(label_file):
    """
    Convert the original (node [list of labels]) label format into k-encoded label
    """
    label_dict = dict()
    label_set = set()

    # Read text label file to dictionary
    with open(label_file) as handle:
        for line in handle.readlines():
            line = line.replace("\n", "")
            split_line = line.split(' ')  # space delimited
            label_dict[int(split_line[0])] = split_line[1:]
            label_set = label_set.union(split_line[1:])

    # Encode labels as 1 X K (K = cardinality of label set) vectors with ith index set for ith class
    labels_encoded = dict()

    for key in label_dict.keys():
        temp_array = np.zeros([len(label_set), ])
        indices = np.array(label_dict.get(key)).astype(int) - 1
        temp_array[indices] = 1
        labels_encoded[key] = temp_array

    # Convert to dataframe of dimension K X N
    k_encoded = pd.DataFrame(labels_encoded)
    # this is just safe code dictionaries maintain insertion order since python 3.6
    encoded_sorted = k_encoded.reindex(sorted(k_encoded.columns), axis=1)

    return encoded_sorted


def emb_file_to_df(emb_file):
    """
    Read emb file to dictionary with key: node id, value: 1 X D vector in embedded space
    """
    data = dict()

    with open(emb_file) as handle:
        first_line = handle.readline()
        num_of_nodes = first_line.split(" ")[0]
        num_of_nodes = int(num_of_nodes)

        data = dict()
        for node_num in range(num_of_nodes):
            line = handle.readline()
            temp_data = []
            for val in line.split(" "):
                temp_data.append(float(val))
            data[int(temp_data[0])] = np.array(temp_data[1:])

    # Convert to D X N dataframe
    patterns = pd.DataFrame(data)
    # safe code for non python 3.6+
    patterns_sorted = patterns.reindex(sorted(patterns.columns), axis=1)

    return patterns_sorted


