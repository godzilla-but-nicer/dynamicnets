import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import os
import time
from tqdm import tqdm
from itertools import chain, combinations


class NetworkSampler:
    def __init__(self, nodes, sample_sizes):
        """This class contains some useful functions for building our graphs"""
        self.data_dir = 'data/edgelists/'
        self.nodes = nodes
        # dictionary of sample sizes so we an easily refer to specific ones
        # make keys be number of nodes and values be sample sizes
        if isinstance(sample_sizes, int):
            self.sample_sizes = {
                n: sample_sizes for n in range(min_nodes, max_nodes + 1)}
        else:
            self.sample_sizes = {
                n: s for n, s in zip(range(min_nodes, max_nodes + 1),
                                     sample_sizes)}

    # from itertools docs
    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r)
                                   for r in range(1, len(s)+1))

    def filter_isomorphisms(self, graphs):
        # This code is taken from stack overflow
        # https://stackoverflow.com/questions/46999771/comparing-a-large-number-of-graphs-for-isomorphism
        unique = []  # A list of unique graphs

        for new in graphs:
            for old in unique:
                if iso.is_isomorphic(new, old):
                    break
            else:
                unique.append(new)

        return unique

    def enumerate_graphs(self, N):
        """enumerate weakly connected directed graphs with N nodes"""
        # easy way to generate an edgelist for the fully connected version
        fully_connected = nx.complete_graph(N, nx.DiGraph)
        full_edgelist = fully_connected.edges()

        possible_edgelists = self.powerset(full_edgelist)
        # we will use this list of graphs to filter out isomorphic graphs
        graphs = []
        for e in possible_edgelists:
            # make new network from edgelist subset
            check = nx.DiGraph()
            check.add_nodes_from(list(range(N)))
            check.add_edges_from(e)

            if nx.is_weakly_connected(check):
                graphs.append(check)

        return graphs

    def sample_graphs(self, N):  # this needs a test for connectivity
        """ Sample connected unlabelled DiGraphs without enumerating them """
        # randomly draw a number of edges
        min_edges = N - 1
        max_edges = N * (N - 1)
        m_edges = np.random.randint(min_edges, max_edges + 1)

        # Construct a list of sampled graphs
        graphs = []
        for g in range(self.sample_sizes[N]):
            # we're going to build up a colection of edges with m = m_edges
            edge_list = []
            for e in range(m_edges):
                # Ensure that the network is connected
                # can do this by forcing a connection for each edge
                if e < N and m_edges >= N:
                    target = np.random.randint(0, N)
                    tup = (e, target)
                    # we can flip this edge with some probability
                    if np.random.uniform() < 0.5:
                        edge = tup[::-1]
                    else:
                        edge = tup

                # there is a case where I draw m = N-1 and I need to ensure
                # that my graph is still connected
                elif e == N - 1 and m_edges == N - 1:
                    source = N
                    target = np.random.randint(0, N)
                    tup = (source, target)
                    if np.random.uniform() < 0.5:
                        edge = tup[::-1]
                    else:
                        edge = tup

                # once we have all of our connections we can just draw randomly
                else:
                    source = np.random.randint(0, N)
                    potential_targets = [t for t in range(N) if t != source]
                    target = np.random.choice(potential_targets)
                    edge = (source, target)

                edge_list.append(edge)

            # use the edgelist to build a graph
            G = nx.DiGraph()
            G.add_nodes_from(list(range(N)))
            G.add_edges_from(edge_list)
            graphs.append(G)

        return graphs

    def write_edgelists(self):
        """ Write out all of the generated Graph's edges to edgelist files """
        n = self.nodes
        start_time = time.time()
        print('generating edgelists for', n, 'nodes')
        if n < 5:
            graphs = self.enumerate_graphs(n)
        else:
            graphs = self.sample_graphs(n)

        graphs = self.filter_isomorphisms(graphs)

        # Check if directory for storing these lists exists
        dir_name = self.data_dir + str(n) + '_node_edgelists/'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # write edgelists to file
        for i, G in enumerate(graphs):
            file_name = dir_name + str(n) + '_node_' + str(i) + '.edgelist'
            fout = open(file_name, 'wb')
            nx.write_edgelist(G, fout)
            fout.close()
        end_time = time.time()
        print('Wrote {} edgelists in {:.2f} s'.format(
            len(graphs), end_time - start_time))
