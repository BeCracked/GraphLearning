import pickle
import time
import warnings
from functools import partial

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score

from multiprocessing import Pool

from Kernels.wl_kernel import wl_kernel
from Kernels.graphlet_kernel import run_graphlet_kernel

from Kernels.closed_walk_kernel import run_cl_kernel

from typing import Tuple, Literal

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
    """
    kern: passes the kernel used to compute the gram matrix as a callable function
    graph_dataset: passes the graph dataset as a list of graphs
    return: the gram matrix as a numpy object
    """
    print(f"Calculating {len(graph_dataset)} feature vectors...")
    kern_start = time.perf_counter()
    feature_vectors = kern(*graph_dataset)
    feature_vectors = [fv / np.sqrt(np.sum(fv ** 2)) for fv in feature_vectors]  # Normalize
    kern_end = time.perf_counter()
    print(f"Calculated feature vectors in {kern_end - kern_start:.4f}s")

    print(f"Constructing gram matrix...")
    pool = Pool()
    # Split work along rows
    graph_start = time.perf_counter()
    gram_m = np.zeros((len(graph_dataset), len(graph_dataset)))
    f = partial(compute_gram_row_chunk, feature_vectors)
    row_chunks = pool.map(f, range(len(feature_vectors)))
    pool.close()
    graph_end = time.perf_counter()

    for i, row_chunk in row_chunks:
        gram_m[i] = row_chunk

    print(f"Took {graph_end-graph_start:.2f}s")

    for m in range(1, len(graph_dataset)):
        for n in range(0, m):
            gram_m[m][n] = gram_m[n][m]

    save_file(gram_m, "gram_matrix")
    return gram_m


def compute_gram_row_chunk(feature_vectors: list, i: int,
                           col_min: int = None, col_max: int = None) -> Tuple[int, np.array]:
    """
    Only calculates the upper right half for any given row.
    Parameters
    ----------
    i : The index of the row to process.
    feature_vectors :
    col_min : Start point in the column.
    col_max : End point in the column.

    Returns Tuple of row index and the complete row with lower left values as 0.
    -------

    """
    if not col_min:
        col_min = i
    if not col_max:
        col_max = len(feature_vectors)
    j_min, j_max = col_min, col_max
    fv_1: csr_matrix = feature_vectors[i]
    gram_row = np.zeros(len(feature_vectors))
    for j in range(j_min, j_max):
        fv_2: csr_matrix = feature_vectors[j]
        d = fv_1 - fv_2
        dist = d.transpose().dot(d).toarray()[0][0]
        gram_row[j] = dist

    return i, gram_row


def compute_gram_matrix_sync(feature_vectors):
    gram_m = np.zeros((len(feature_vectors), len(feature_vectors)))
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
            print(f"Outer: {i}/{len(feature_vectors)} ({i/len(feature_vectors)*100:.2f}%) "
                  f"(Total time: {outer_end-outer_start:.2f}s)")

    print(f"Gram matrix construction took {time.perf_counter() - outer_start:.2f}s")


def fit(kern: Literal["closed_walk", "graphlet", "WL"], dataset: Literal["DD", "Enzymes", "NCI"]):
    """
    kern: gives a kernel to fit an svm to the dataset passed
    dataset: The dataset based on which the gram matrix is computed (i.e. to which the svm is fitted)
    Gives no return value but prints the accuracy and std for the given kernel on the given dataset
    """
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
            print(f"Error: {dataset} is not a valid dataset")

    data = data[:int(len(data) / 1)]  # TEMP: Reduce dataset size

    kern_func = None
    match kern:
        case "WL":
            kern_func = partial(wl_kernel, 4)
        case "closed_walk":
            l = 100
            kern_func = partial(run_cl_kernel, l)
        case "graphlet":
            kern_func = partial(run_graphlet_kernel)
            data = [g for g in data if len(g) >= 5]  # Drop too small graphs from data
        case _:
            print(f"Error: {kern} is not a valid kernel")
    print(f"Fitting SVMs on dataset {dataset!r} with kernel {kern!r}")
    fit_start = time.perf_counter()
    gram_matrix = compute_gram_matrix(kern_func, data)
    labels = extract_labels_from_dataset(data)
    clf = svm.SVC(kernel='precomputed', random_state=42)
    scores = cross_val_score(clf, gram_matrix, labels, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    fit_end = time.perf_counter()
    print(f"Fitting took {fit_end - fit_start:.2f}s")


if __name__ == '__main__':
    fit("closed_walk", "DD")
