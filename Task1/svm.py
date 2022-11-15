import pickle
import time
import warnings
from collections import defaultdict
from functools import partial
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
import argparse as ap

from Kernels.closed_walk_kernel import run_cl_kernel
from Kernels.graphlet_kernel import run_graphlet_kernel
from Kernels.wl_kernel import wl_kernel

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_file(dict_name: str):
    with open(dict_name, 'rb') as f:
        loaded_file = pickle.load(f)
        return loaded_file


def extract_labels_from_dataset(dataset):
    return [element.graph['label'] for element in dataset]


class SVCGramEstimator(BaseEstimator):
    """
    Wraps a precomputed SVC estimator by computing the gram matrix
    """

    def __init__(self):
        self.clf = SVC(kernel="precomputed")
        self.x_trained = None

    def fit(self, x, y):
        self.x_trained = x
        gram = (x @ x.T)
        # Ensure gram matrix is dense
        if not isinstance(gram, np.ndarray):
            gram = gram.toarray()

        self.clf.fit(gram, y)

    def predict(self, x):
        gram = (x @ self.x_trained.T)
        # Ensure gram matrix is dense
        if not isinstance(gram, np.ndarray):
            gram = gram.toarray()

        return self.clf.predict(gram)


def fit(kern: Literal["closed_walk", "graphlet", "WL"] | str,
        dataset: Literal["DD", "Enzymes", "NCI"] | str,
        normalize: bool = False):
    """
    kern: gives a kernel to fit an svm to the dataset passed
    dataset: The dataset based on which the gram matrix is computed (i.e. to which the svm is fitted)
    Gives no return value but prints the accuracy and std for the given kernel on the given dataset
    """
    print("##########################################################################\n")
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

    kern_func = None
    match kern:
        case "WL":
            kern_func = partial(wl_kernel, 4)
        case "closed_walk":
            l = 100
            normalize = True
            kern_func = partial(run_cl_kernel, l)
        case "graphlet":
            kern_func = partial(run_graphlet_kernel)
            data = [g for g in data if len(g) >= 5]  # Drop too small graphs from data
        case _:
            print(f"Error: {kern} is not a valid kernel")

    print(f"Fitting SVMs on dataset {dataset!r} with kernel {kern!r}")
    fit_start = time.perf_counter()
    # Get embeddings based on kernel feature vectors
    kern_start = time.perf_counter()
    fvs = kern_func(*data)
    kern_end = time.perf_counter()
    print(f"Calculated feature vectors in {kern_end - kern_start:.4f}s")

    # Gather labels (aka. y)
    labels = extract_labels_from_dataset(data)

    # Shuffle data
    x, y = shuffle(fvs, labels, random_state=42)

    # Normalize if flag set
    if normalize:
        scaler = StandardScaler(with_mean=False)
        scaler.fit(x)
        x = scaler.transform(x)

    clf = SVCGramEstimator()
    scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')

    print(f"Fitting overall took {time.perf_counter() - fit_start:.2f}s")
    print(f"Accuracy: {scores.mean():.2f}±{scores.std():.2f}")

    return scores.mean(), scores.std()


def run_all():
    kernels = [
        "WL",
        "closed_walk",
        "graphlet"
    ]
    results = defaultdict(dict)
    datasets = ["Enzymes", "NCI", "DD"]
    for ds in datasets:
        for kernel in kernels:
            results[ds][kernel] = fit(kernel, ds)

    print(results)


def run():
    parser = ap.ArgumentParser(prog="Fit SVM",
                               description="Trains an SVM based on the passed kernel function and dataset")
    parser.add_argument("kernel", choices=["closed_walk", "graphlet", "WL"])
    parser.add_argument("dataset_name", choices=["DD", "Enzymes", "NCI"])

    args = parser.parse_args()

    result = fit(args.kernel, args.dataset_name)
    print(result)


if __name__ == '__main__':
    run()
