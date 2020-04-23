import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm
from itertools import chain, combinations

# from itertools docs
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def generate_networks(N):
    """generate edgelists for all weakly connected directed graphs with N nodes"""
    # easy way to generate an edgelist for the fully connected version
    fully_connected = nx.complete_graph(N, nx.DiGraph)
    full_edgelist = fully_connected.edges()

    possible_edgelists = powerset(full_edgelist)
    # we will use this list of graphs to filter out isomorphic graphs
    G_list = []
    for e in possible_edgelists:
        # make new network from edgelist subset
        check = nx.DiGraph()
        check.add_nodes_from(list(range(N)))
        check.add_edges_from(e)

        if nx.is_weakly_connected(check):
            G_list.append(check)

    # loop over all combinations of graphs to check for isomorphisms
    # hopefully modifying like this isn't going to make everything explode
    # This doesn't get me to the actual number of graphs but it is much lower than
    # The unfiltered number
    for graphi in G_list:
        for j, graphj in enumerate(G_list):
            if iso.is_isomorphic(graphi, graphj) and graphi != graphj:
                G_list.pop(j)
    
    return G_list