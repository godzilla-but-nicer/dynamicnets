import networkx as nx
import numpy as np


class ThresholdBooleanNetwork:
    def __init__(self, edgelist):
        """ Boolean Networks are built from edgelists and consist of the node and
        edge objects defined elsewhere.

        Parameters
        ----------
        edgelist: str
            string containing the relative path to a file containing a networkx
            style edgelist
        """
        # load the networkx graph representation
        self.graph = nx.read_edgelist(edgelist)

        # Track node states and thresholds with vectors
        self.states = np.ones(len(self.graph.nodes()), dtype=int)
        self.thresholds = np.zeros(self.states.shape[0], dtype=int)

        # track connections and weights with a matrix
        adj_matrix = np.array(nx.to_numpy_matrix(self.graph), dtype=int)
        np.fill_diagonal(adj_matrix, 1)  # self edges
        weights = np.random.randint(-5, 5, size=(adj_matrix.shape))  # initial weights
        self.weight_matrix = adj_matrix * weights

        # used to track changes in time
        self.time = 0

    def print_node_states(self):
        """ Print out the current state of each node """
        print('\nt =', self.time)
        print('Node\tin_degree\tstate')
        print('-'*30)
        for id in range(len(self.states)):
            state = self.states[id]
            in_degree = np.count_nonzero(self.weight_matrix[:, id])
            print('{}\t{}\t\t{}'.format(id, in_degree, state))

        return True

    def synchronous_step(self):
        """ Step forward in time, calculating transitions for each node """
        signal_matrix = self.states * self.weight_matrix
        signals = np.sum(signal_matrix, axis=1)
        self.states = (self.thresholds < signals).astype(int)

        self.time += 1

        return True
