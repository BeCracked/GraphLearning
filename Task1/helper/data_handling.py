import pickle
from typing import Iterable


def extract_labels_from_dataset(dataset: Iterable) -> list:
    return [element.graph['label'] for element in dataset]


def load_file(dict_name: str):
    with open(dict_name, 'rb') as f:
        loaded_file = pickle.load(f)
        return loaded_file
