import pickle
import time
import warnings
from functools import partial

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score

from Kernels.wl_kernel import wl_kernel

from typing import Literal

warnings.filterwarnings("ignore", category=DeprecationWarning)


def save_file(a_dict, filename):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(a_dict, f)


def load_file(dict_name: str):
    with open(dict_name, 'rb') as f:
        loaded_file = pickle.load(f)
        return loaded_file


def extract_labels_from_dataset(dataset):
    return [element.graph['label'] for element in dataset]


def compute_gram_matrix(kern: callable, graph_dataset):
    gram_m = np.zeros((len(graph_dataset), len(graph_dataset)))
    print(f"Calculating {len(graph_dataset)} feature vectors...")
    kern_start = time.perf_counter()
    feature_vectors = kern(*graph_dataset)
    kern_end = time.perf_counter()
    print(f"Calculated feature vectors in {kern_end-kern_start:.4f}s")

    print(f"Constructing gram matrix...")
    outer_start = time.perf_counter()
    for i in range(0, len(feature_vectors)):
        fv_1: csr_matrix = feature_vectors[i]
        # as the gram matrix will be symmetric, we only compute the upper half and copy it to the button half later on
        for j in range(i, len(feature_vectors)):
            fv_2: csr_matrix = feature_vectors[j]
            d = fv_1 - fv_2
            dist = d.transpose().dot(d).toarray()[0][0]
            gram_m[i][j] = dist

        if i % 100 == 0:
            outer_end = time.perf_counter()
            print(f"Outer: {i}/{len(graph_dataset)} ({i/len(graph_dataset)*100:.2f}%) (Total time: {outer_end-outer_start:.2f}s)")

    print(f"Gram matrix construction took {time.perf_counter() - outer_start:.2f}s")
    for m in range(1, len(graph_dataset)):
        for n in range(0, m):
            gram_m[m][n] = gram_m[n][m]

    save_file(gram_m, "gram_matrix")
    return gram_m


def fit(kern: Literal["closed_walk", "graphlet", "WL"], dataset: Literal["DD", "Enzymes", "NCI"]):
    dataset_dir = "./datasets"
    data = None
    match dataset:
        case "DD":
            data = load_file(f"{dataset_dir}/DD/data.pkl")
        case "Enzymes":
            data = load_file(f"{dataset_dir}/ENZYMES/data.pkl")
        case "NCI":
            data = load_file(f"{dataset_dir}/NCI1/data.pkl")
        case _:
            print(f"Error: {kern} is not a valid kernel")

    data = data[:int(len(data)/1)]  # TEMP: Reduce dataset size

    clf = svm.SVC(kernel='precomputed', random_state=42)
    kern_func = None
    match kern:
        case "WL":
            kern_func = partial(wl_kernel, 4)
        case _:
            print(f"Error: {kern} is not a valid kernel")
    fit_start = time.perf_counter()
    gram_matrix = compute_gram_matrix(kern_func, data)
    labels = extract_labels_from_dataset(data)
    scores = cross_val_score(clf, gram_matrix, labels, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    fit_end = time.perf_counter()
    print(f"Fitting took {fit_end-fit_start:.2f}s")


if __name__ == '__main__':
    fit("WL", "DD")
