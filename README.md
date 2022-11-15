# Graph Learning
Please use this readme to describe how to run your code and the results you obtained in each exercise.

## Repository Structure

The ``requirements.txt`` can be installed with the following command:

``pip install -r requirements.txt``

Everything else that is relevant for the first exercise can be found in folder ``Task1``:

* ``Kernels`` folder: contains all three kernels (Closed Walk, Graphlets and Weisfeiler-Leman) and ``graphlets.py`` which is a collection of all 34 graphlets.
* ``datasets`` folder: contains the given three datasets (DD, ENZYMES and NCI1).
* ``helper`` folder: contains some methods to process the input datasets and generate graphs to test the kernels.
* ``run.py`` file: is just for testing the correctness of the kernels and is not used during the SVM training.
* ``svm.py`` file: to calculate the gram matrix and train/evaluate the SVM model on all 3 datasets using all 3 kernels (main execution file).

## How To Run Scripts

### To train an SVM and print the evaluation results obtained using 10-fold cross validation

Go to folder ``Task1`` and execute the following commands:

``python svm.py closed_walk DD`` 

``python svm.py closed_walk Enzymes``

``python svm.py closed_walk NCI``

``python svm.py graphlet DD``

``python svm.py graphlet Enzymes``

``python svm.py graphlet NCI``

``python svm.py WL DD``

``python svm.py WL Enzymes``

``python svm.py WL NCI``

### To test the kernels and print the feature vectors based on a kernel and given graphs
Go to folder ``Task1`` and execute the following commands:

``python run.py closed_walk <graphs>``

``python run.py graphlet <graphs>``

``python run.py WL <graphs>``

 where ``<graphs>`` is a list of adjacency matrices as strings.

An example command to execute Weisfeiler-Lehman-Kernel on two graphs defined by two adjacency matrices.

```bash
python run.py WL "[[0,1,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]" "[[0,1,1,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]"
```

## Evaluation Results

|         | Graphlet kernel | Closed Walk kernel | Weisfeiler-Lehman kernel |
|---------|-----------------|--------------------|--------------------------|
| DD      | 74.35 (±3.58)   | 74.79 (±2.82)      | 78.61 (±2.45)            |
| ENZYMES | 25.28 (±3.82)   | 22.5 (±6.69)       | 53.67 (±6.72)            |
| NCI1    | 62.42 (±2.32)   | 65.35 (±1.90)      | 84.67 (±1.12)            |

## Comparison with Paper Results

### Excerpt of the Accuracy Results in the Paper
|         | Graphlet count kernel | p-random walk kernel | WL subtree  kernel  |
|---------|----------------|---------------|---------------|
| DD      | 78.59 (±0.12)  | 66.64 (±0.83) | 79.78 (±0.36) |
| ENZYMES | 32.70 (±1.20)  | 27.67 (±0.95) | 46.42 (±1.35) |
| NCI1    | 66.00 (±0.07)  | 58.66 (±0.28) | 82.19 (±0.18) |

The walk-based kernels used in the paper, (p-)random walk, use random walks which is why it is not sensible to compare them to the closed walk kernel we implemented. For the graphlet kernel, the kernel from the paper achieved slightly better accuracies and for the Weisfeiler-Lehman kernel we achieved similar accuracies on DD and NCI1 and a much better result on ENZYMES. The standard deviations of the results of the paper are a lot smaller than ours. This might be because we consider the deviations directly on the score itself while the paper most likely gives the error of the estimated mean.

### Comment: Number of closed walks of length k (Closed Walk Kernel)

According to literature (https://users.monash.edu/~gfarr/research/slides/Minchenko-RSPlanTalk.pdf), the number of closed walks of length k in G equals the trace of the adjacency matrix of G which is exponentiated by k (it considers the eigenvalues of the adjacency matrix of G). We used this formula for the closed walk kernel.


### Comment: Choice of maximal length l (Closed Walk Kernel)

Since the formula we used considers the eigenvalues of a given graph G, which is dependent on the number of nodes in G, we investigated how large the graphs in the datasets are. We observed that some are very small (two to more than a hundred nodes) and some have over thousand nodes. Since the size of the graphs vary a lot, we decided in the end to set the maximal length of the closed walks to 100.

