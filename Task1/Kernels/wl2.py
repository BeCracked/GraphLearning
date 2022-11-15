import numpy as np
import pickle
import networkx as nx
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from random import sample, shuffle
import os
import scipy.sparse

# The following are two helper variables:
my_own_hash_dic = dict()  # which saves a mapping from a nodes current color and the colors of its neighbors to a new color
start_colors = 1  # the amount of initial colors in a graph


def get_Hash_val(currentNodeColor, currentNeighborColors):
    """
    returns a color, depending on the current node color and the color of its neighbors
    if there is no entry for this case in our hash function yet, we create one
    :param currentNodeColor: the current color of the node (as an int)
    :param currentNeighborColors: the current colors of the neighbors (as a str which contains the colors ordered)
    :return: the new hash/color for a node
    """
    if (currentNodeColor, currentNeighborColors) in my_own_hash_dic:
        return my_own_hash_dic[(currentNodeColor, currentNeighborColors)]
    else:
        curr_len = len(my_own_hash_dic)
        my_own_hash_dic[(currentNodeColor, currentNeighborColors)] = curr_len + start_colors
        return curr_len


def refine_coloring(graphs, currentcolors):
    """
    performs one iteration of color refinement on the passed graphs with the current coloring
    :param graphs: list of graphs whose coloring we want to refine
    :param currentcolors: list of lists which contains an entry for every node in every graph saving its current color
    :return: new list of lists containing the updated colors
    """
    new_colors = [None] * len(graphs)
    for i in range(len(graphs)):
        new_colors[i] = dict()
        for node in graphs[i].nodes:
            curr_col = currentcolors[i][node]
            neighbor_cols = []
            for n in graphs[i].neighbors(node):
                neighbor_cols.append(currentcolors[i][n])
            neighbor_cols.sort()
            new_colors[i][node] = get_Hash_val(curr_col, str(neighbor_cols))
    return new_colors


def init_coloring(graphs):
    """
    Creates an initial coloring for the nodes in the graphs.
    We set the initial colors to the node_labels of the nodes, if there are none, we choose a default color
    :param graphs: initial list of graphs whose nodes we want to color
    :return: list of dicts, which contains for every node of every graph its initial color
    """
    colors = [None] * len(graphs)
    for i in range(len(graphs)):
        colors[i] = dict()
        for (node, data) in zip(graphs[i].nodes, graphs[i].nodes(data=True)):
            colors[i][node] = 0  # if no node label
            if "node_label" in data[1]:
                colors[i][node] = data[1]["node_label"]
    temp_set = set()
    for i in range(len(graphs)):
        for node in graphs[i].nodes:
            temp_set.add(colors[i][node])
    start_colors = len(temp_set)

    return colors


def get_Color_Hist(current_colors):
    """
    creates a list which contains every color that occurs in the current coloring of all graphs
    :param current_colors: list of dicts which contains for every node in every graph a color
    :return: list which contains every color exactly once if it occurs in the list of lists
    """
    color_set = set()
    for i in range(len(current_colors)):
        temp = set.union(set(current_colors[i].values()), color_set)
        color_set = temp
    return list(color_set)


def get_color_dist(current_colors):
    """
    :param current_colors: list of dicts which contains for every node of every graph its current color
    :return: list of (sparse) vectors which indicate how many nodes of a certain color every graph has
    """
    color_list = get_Color_Hist(current_colors)
    print("there are " + str(len(color_list)) + " different colors.")
    res = [None] * len(current_colors)
    for i in range(len(current_colors)):
        temp = list(current_colors[i].values())
        counts = dict()
        for j in temp:
            counts[j] = counts.get(j, 0) + 1
        temp_k = []
        count = 0
        for color in color_list:
            temp_k.append(counts.get(color, 0))
        res[i] = scipy.sparse.csr_matrix([temp_k])
    return res


def get_WL_Vectors(iterations, *graphs, **kwargs):
    """
    transforms the input graphs to vectors embedding the WL coloring
    :param graphs: list of graphs we want to transform to vectors
    :param iterations: the number of iterations the WL algorithm shall run
    :return: the sparse vectors encoding the colorings
    """
    graphs = list(graphs)
    res_colors = [None] * len(graphs)
    for i in range(len(graphs)):  # initialize vectors
        res_colors[i] = scipy.sparse.csr_matrix((1, 0))

    init_colors = init_coloring(graphs)  # perform initial coloring
    init_color_dis = get_color_dist(init_colors)
    for i in range(len(graphs)):
        res_colors[i] = scipy.sparse.hstack([res_colors[i], init_color_dis[i]])

    colors = init_colors  # perform remaining coloring iterations
    for i in range(1, iterations + 1):
        colors = refine_coloring(graphs, colors)
        color_dis = get_color_dist(colors)
        for j in range(len(graphs)):
            res_colors[j] = scipy.sparse.hstack([res_colors[j], color_dis[j]])
    return scipy.sparse.vstack(res_colors)


