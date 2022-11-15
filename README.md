# Graph Learning
Please use this readme to describe how to run your code and the results you obtained in each exercise.

## Repository Structure
Everything that is relevant for the first exercise can be found in folder ``Task1``:

* ``Kernels`` folder: contains all three kernels (Closed Walk, Graphlets and Weisfeiler-Leman) and ``graphlets.py`` which is a collection of all 34 graphlets.
* ``datasets`` folder: contains the given three datasets (DD, ENZYMES and NCI1).
* ``helper`` folder: contains some methods to process the input datasets and generate graphs to test the kernels.
* ``run.py`` file: to execute all kernels and obtain the feature vectors
* ``svm.py`` file: to calculate the gram matrix and train/evaluate the SVM model on all 3 datasets, using all 3 kernels

## How To Run Scripts

### To train an SVM and print the evaluation results obtained using 10-fold cross validation
``python svm.py Kernel Dataset`` where Kernel is either `closed_walk`, `graphlet`, or `WL` and Dataset is either `DD`, `Enzymes`, or `NCI`

### To construct and print the feature vectors based on a kernel and a graph dataset
``python run.py Kernel Graphs`` where Kernel is either `closed_walk`, `graphlet`, or `WL` and Graphs is a list of adjacency matrices as strings.

An example command to execute Weisfeiler-Lehman-Kernel on two graphs defined by two adjacency matrices.
```bash
python Task1/run.py WL "[[0,1,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]" "[[0,1,1,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]"
```

## Evaluation Results

|         | Graphlet kernel | Closed Walk kernel | Weisfeiler-Lehman kernel |
|---------|-----------------|--------------------|--------------------------|
| DD      | 74.35 (±3.58)   | 74.79 (±2.82)      | 78.61 (±2.45)            |
| ENZYMES | 25.28 (±3.82)   | 22.5 (±6.69)       | 53.67 (±6.72)            |
| NCI1    | 62.42 (±2.32)   | 65.35 (±1.90)      | 84.67 (±1.12)            |

## Comparison with Paper Results
The walk-based kernels used in the paper, (p-)random walk, use random walks which is why it's not sensible to compare to the closed walk kernels we implemented.
The accuracies are better everywhere except for Closed Walk/p-Random Walk and WL kernel on DD and NCI1.

The standard deviations of the results of the paper are a lot smaller than ours. This might be because we consider the error on the score itself while the paper gives the error of the estimated mean.

### Excerpt of the Accuracy Results in the Paper
|         | Graphlet count | p-random walk | WL subtree    |
|---------|----------------|---------------|---------------|
| DD      | 78.59 (±0.12)  | 66.64 (±0.83) | 79.78 (±0.36) |
| ENZYMES | 32.70 (±1.20)  | 27.67 (±0.95) | 46.42 (±1.35) |
| NCI1    | 66.00 (±0.07)  | 58.66 (±0.28) | 82.19 (±0.18) |

### Comment: Choice of maximal length l (Closed Walk Kernel)

### Comment: Number of closed walks of length k (Closed Walk Kernel)

According to references (https://users.monash.edu/~gfarr/research/slides/Minchenko-RSPlanTalk.pdf), the number of closed walks of length k in G equals the trace of the adjacency matrix of G which is exponentiated by k.

