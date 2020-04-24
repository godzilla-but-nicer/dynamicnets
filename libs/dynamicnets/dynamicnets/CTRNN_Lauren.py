# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:49:15 2019

@author: Lauren Benson
"""

import numpy as np
import networkx as nx


def sigmoid(x):
    return 1/(1+np.exp(-x))


def inv_sigmoid(x):
    return np.log(x / (1-x))


class CTRNN():

    def __init__(self, size, edgelist):
        self.Size = size                        # number of neurons
        self.Voltage = np.zeros(size)           # neuron activation vector
        self.TimeConstant = np.ones(size)       # time-constant vector
        self.Bias = np.zeros(size)              # bias vector
        self.Weight = np.zeros((size, size))     # weight matrix
        self.Output = np.zeros(size)            # neuron output vector
        self.Input = np.zeros(size)             # neuron output vector
        self.AdjFilter = self.get_adjacency_filter(edgelist)  # restrict conns.

    def randomizeParameters(self):
        self.Weight = np.random.uniform(-10, 10, size=(self.Size, self.Size))
        self.Bias = np.random.uniform(-10, 10, size=(self.Size))
        self.TimeConstant = np.random.uniform(0.1, 5.0, size=(self.Size))
        self.invTimeConstant = 1.0/self.TimeConstant

    def setParameters(self, genotype, WeightRange, BiasRange,
                      TimeConstMin, TimeConstMax):
        k = 0
        for i in range(self.Size):
            for j in range(self.Size):
                self.Weight[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):
            self.Bias[i] = genotype[k]*BiasRange
            k += 1
        for i in range(self.Size):
            self.TimeConstant[i] = ((genotype[k] + 1)/2) * \
                (TimeConstMax-TimeConstMin) + TimeConstMin
            k += 1
        self.invTimeConstant = 1.0/self.TimeConstant

    def initializeState(self, v):
        self.Voltage = v
        self.Output = sigmoid(self.Voltage+self.Bias)

    def initializeOutput(self, o):
        self.Output = o
        self.Voltage = inv_sigmoid(o) - self.Bias

    def step(self, dt):
        netinput = self.Input + np.dot(self.Weight.T, self.Output)
        self.Voltage += dt * (self.invTimeConstant*(-self.Voltage+netinput))
        self.Output = sigmoid(self.Voltage+self.Bias)

    def setweightmatrix3(self, numconnections):
        self.Weight = np.random.uniform(-15, 15, size=(self.Size, self.Size))
        self.Weight = self.Weight * self.AdjFilter

    def get_adjacency_filter(self, edgelist):
        # load the edge list into a graph and return adjacency matrix
        G = nx.read_edgelist(edgelist)
        adj_matrix = np.array(nx.to_numpy_matrix(G))
        adj_filter = np.fill_diagonal(adj_matrix, 1)
        return adj_filter
