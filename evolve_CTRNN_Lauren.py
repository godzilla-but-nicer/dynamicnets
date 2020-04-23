# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:55:16 2019

@author: Lauren Benson
"""

import numpy as np
import CTRNN_Lauren
import matplotlib.pyplot as plt

#Neural network parameters
size = 100
duration = 100
stepsize = 0.05
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0
TimeConstMax = 2.0

time = np.arange(0.0,duration,stepsize)

#fitness function

def fitnessFunction(genotype):
    nn = CTRNN_Lauren.CTRNN(size)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    fitness_score = 0.0
    trials = 0
    nn.initializeState(np.zeros(size))
    for t in time:
        pastOutputs = nn.Output
        nn.step(stepsize)
        currentOutputs = nn.Output
        fitness_score += np.sum(abs(currentOutputs - pastOutputs)/stepsize)
    trials += 1
    return fitness_score/(size*fitduration*trials)

#Evolutionary Algorithm parameters
    
popsize = 100
genesize = 10
recombProb = 0.75
mutatProb = 0.01
generations = 50
tournaments = generations * popsize
repetitions = 5

#Evolve fitness over generations


