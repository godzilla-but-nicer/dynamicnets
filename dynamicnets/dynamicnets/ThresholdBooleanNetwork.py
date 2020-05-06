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
        self.graph = nx.read_edgelist(edgelist, create_using=nx.DiGraph)

        # Track node states and thresholds with vectors
        self.states = np.ones(len(self.graph.nodes()), dtype=int)
        self.outputs = np.zeros(self.states.shape[0])
        self.thresholds = np.ones(self.states.shape[0], dtype=int)

        # track connections and weights with a matrix
        adj_matrix = np.array(nx.to_numpy_matrix(self.graph), dtype=int)
        np.fill_diagonal(adj_matrix, 1)  # self edges
        # initialize weights
        weights = np.random.randint(-5, 5, size=(adj_matrix.shape))
        self.weight_matrix = adj_matrix * weights
        self.adj_matrix = adj_matrix

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
        signals = np.dot(self.weight_matrix.T, self.states)
        self.states = (self.thresholds < signals).astype(int)
        # we need to track a separate output to make our aggregated 
        # oscillations map onto the CTRNNs
        self.outputs = self.states
        self.outputs[self.states == 0] = -1

        self.time += 1

        return True

    def ga_set_weights(self, genotype, weight_range):
        """ Set the weight matrix using the genotype from the 
        genetic algorithm """
        geno_mat = genotype.reshape(self.weight_matrix.shape)
        self.weight_matrix = geno_mat * weight_range * self.adj_matrix

        return True

