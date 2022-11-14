
import pm4py
import pickle
import networkx


dataset_dd = "datasets/DD/data.pkl"
dataset_en = "datasets/ENZYMES/data.pkl"
dataset_nc = "datasets/NCI1/data.pkl"


def load_file(dict_name: str):
    with open(dict_name, 'rb') as f:
        loaded_file = pickle.load(f)
        return loaded_file


enzymes = load_file(dataset_dd)
print(enzymes)

