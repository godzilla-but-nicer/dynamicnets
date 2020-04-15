import numpy as np
import networkx as nx
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
    keep_edgelists = []

    for e in tqdm(possible_edgelists):
        # make new network from edgelist subset
        check = nx.DiGraph()
        check.add_edges_from(e)

        if nx.is_weakly_connected(check):
            keep_edgelists.append(e)
    
    return keep_edgelists