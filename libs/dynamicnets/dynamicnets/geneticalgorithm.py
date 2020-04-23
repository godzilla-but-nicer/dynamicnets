import numpy as np

# our genotype will be a 3-tensor with shape (PopSize, N, N)
# (N, N) consist of a matrix of link weights


class GA:
    """ Data and functions required for the evolution of our dynamic networks """

    def __init__(self, pop_size, N, mut_rate, rec_rate, deme_size):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.rec_rate = rec_rate
        self.deme_size = deme_size
        self.genotype = np.zeros((pop_size, N, N))