""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""

import numpy as np
from Tree import Tree
from Tree import Node
from collections import defaultdict

def get_child(n, pa):
    ch = []
    for node, parent in enumerate(pa):
        if parent == n:
            ch.append(node)
    return ch

def get_prob(CPT, n, col, parent_col=-1):
    if parent_col == -1:    return CPT[n][col]
    else:                   return CPT[n][parent_col][col]

def get_sib(n, pa):
    for node, parent in enumerate(pa):
        if node != n:
            if np.isnan(parent) and np.isnan(pa[n]):
                return node
            elif pa[n] == parent:
                return node

def calc_s_val(n, col, pa, CPT, beta, storage):
    """
    Recursively calculated S values downwards from the root.
    @params: 
        n   :   current node
        col :   category
        pa  :   tree_topology - pa[i] = parent of node i
        CPT :   conditional probability table
        beta    :   observed values   
        storage :   stores all computed S values
    @return:
        s   : S value of current node and category
    """

    if storage[n, col] != -1:  return storage[n][col]
    
    ch = get_child(n, pa)
    if len(ch) == 0:    # If at leaf node
        storage[n,col] = 1 if int(beta[n]) == col else 0
        return storage[n,col]
    
    results = np.zeros(len(ch))
    for chID, chNode in enumerate(ch):
        for column in range(len(CPT[0])):
            results[chID] += \
                calc_s_val(chNode, column, pa, CPT, beta, storage) \
                    * get_prob(CPT, chNode, column, col)
    
    s = np.prod(results)
    storage[n,col] = s
    return s

def calc_t_val(n, col, pa, CPT, storage, s_vals):
    """
    Recursively calculated T values upwards from the leaf node.
    @params: 
        n   :   current node
        col :   category
        pa  :   tree_topology - pa[i] = parent of node i
        CPT :   conditional probability table
        storage :   stores all computed T values
        s_vals  :   stores all computed S values
    @return:
        t   : T value of current node and category
    """

    if np.isnan(pa[n]):
        return get_prob(CPT, n, col)
    if storage[n,col] != -1:
        return storage[n,col]
    
    sib = get_sib(n, pa)
    t = 0
    for i in range(len(CPT[0])):
        for j in range(len(CPT[0])):
            t += get_prob(CPT, n, col, i) * get_prob(CPT, sib, j, i) \
                    * s_vals[sib,j]  * calc_t_val(int(pa[n]), i, pa, CPT, storage, s_vals)
    
    storage[n,col] = t
    return t

    

def calculate_likelihood(tree_topology, theta, beta, tree):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """

    # TODO Add your code here
    storage_s = np.ones((len(tree_topology), len(theta[0]))) * -1
    storage_t = np.ones((len(tree_topology), len(theta[0]))) * -1

    # Calculate S values for all nodes for all categories
    for col in range(len(theta[0])): calc_s_val(0, col, tree_topology, theta, beta, storage_s)

    # Recursively calculated the T values for all nodes and the likelihood of the sample
    likelihood = 0
    for node, val in enumerate(beta):
        if not np.isnan(val):
            likelihood = \
                calc_t_val(node, int(val), tree_topology, theta, storage_t, storage_s) \
                    * storage_s[node,int(val)]
            break
    
    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    # filename = "data/q2_2/q2_2_small_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    # filename = "data/q2_2/q2_2_medium_tree.pkl" # "data/q2_2/q2_2_large_tree.pkl"
    filename = "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta, t)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
