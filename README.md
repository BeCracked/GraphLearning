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

## Evaluation Results

|          | Closed Walk kernel | Graphlet kernel | Weisfeiler-Leman kernel |
|---       |--------------------|-----------------|-------------------------|
|  DD      | 0.76 (+/- 0.02)    | 0.75 (+/- 0.03) | 0.41 (+/- 0.00)         |
|  ENZYMES | 0.19 (+/- 0.05)    | 0.25 (+/- 0.06) | 0.18 (+/- 0.01)         |
|  NCI1    | 0.64 (+/- 0.02)    | 0.63 (+/- 0.05) | 0.79 (+/- 0.03)         |

## Comparison with Paper Results 

### Comment: Choice of maximal length l (Closed Walk Kernel)

### Comment: Number of closed walks of length k (Closed Walk Kernel)

According to references (https://users.monash.edu/~gfarr/research/slides/Minchenko-RSPlanTalk.pdf), the number of closed walks of length k in G equals the trace of the adjacency matrix of G which is exponentiated by k.

<!---
Example command to execute Weisfeiler-Leman-Kernel on two graphs defined by two adjacency matrices.

```bash
python Task1/run.py WL "[[0,1,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]" "[[0,1,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]"
```
-->

