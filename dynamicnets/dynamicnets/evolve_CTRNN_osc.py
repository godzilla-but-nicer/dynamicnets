# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:21:21 2019

@author: Lauren Benson
"""

from MGA import Microbial
from CTRNNEdgelist import CTRNN
import numpy as np
import matplotlib.pyplot as plt


# datafile where network structure is stored
edgelist = ''
struct_id = edgelist.split('_')[2]  # arbitrary network id stored in name


# NN params
# first couple of chars in edgelist name is the number of nodes
nnsize = int(edgelist.split('_')[0])
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0
TimeConstMax = 2.0

# timescale params
fitduration = 10
duration = 40
transient = 10
stepsize = 0.05

# different things occur on different time scales
fittime = np.arange(0.0, fitduration, stepsize)  # what does this mean?
time = np.arange(0.0, duration, stepsize)
transientdur = np.arange(0.0, transient, stepsize)  # what is this for
totaltime = np.arange(0.0, duration+transient, stepsize)  # what is this for


# The sum of absolute differences in each trace and their past
def fitnessFunction(genotype, edgelist):
    # initialize network and set paramss in some bounds
    nn = CTRNN(edgelist)
    nn.setParameters(genotype, WeightRange, BiasRange,
                     TimeConstMin, TimeConstMax)

    # calculate fitness over the fitness timescale
    fit = 0.0
    trials = 0
    nn.initializeState(np.zeros(nnsize))
    for t in fittime:
        pastOutputs = nn.Output
        nn.step(stepsize)
        currentOutputs = nn.Output
        fit += np.sum(abs(currentOutputs - pastOutputs)/stepsize)
    trials += 1
    return fit/(nnsize * fitduration * trials)


# EA params
popsize = 10
genesize = nnsize*nnsize + 2*nnsize
recombProb = 0.5
mutatProb = 0.1
generations = 50
tournaments = generations * popsize
osc_strength = 5.0

# Repeat evolution to find circuits where:
#   1. Oscillations are strongly enough
#   2. Each individual trace is periodic
#   3. activities sum above a threshold
for r in range(10):
    # Evolve and visualize fitness over generations
    ga = Microbial(fitnessFunction, popsize,
                   genesize, recombProb, mutatProb)
    ga.run(tournaments)
    ga.showFitness()
    # Get best evolved network and show its activity
    afit, bfit, bi = ga.fitStats()

    # First condition: That the oscillation be strong
    if bfit > osc_strength:
        # init ctrnn
        nn = CTRNN(edgelist)
        nn.setParameters(bi, WeightRange, BiasRange,
                         TimeConstMin, TimeConstMax)
        nn.initializeState(np.zeros(nnsize))
        #
        outputs = np.zeros((len(totaltime), nnsize))
        summed = np.zeros(len(totaltime))
        lastpeak = np.zeros(nnsize)-1
        periodvar = np.zeros(nnsize)
        peaklist = [[], [], []]  # what is this
        step = 0
        for t in transientdur:
            nn.step(stepsize)
            outputs[step] = nn.Output
            summed[step] = np.mean(nn.Output)
            step += 1
        summedactivity = 0.0
        for t in time:
            summedPastOutputs = np.sum(nn.Output)
            nn.step(stepsize)
            summedCurrentOutputs = np.sum(nn.Output)
            summedactivity += np.sum(abs(summedCurrentOutputs -
                                         summedPastOutputs)/stepsize)
            outputs[step] = nn.Output
            summed[step] = np.mean(nn.Output)
            # Find period for each neuron
            for i in range(nnsize):
                if (outputs[step-1][i] > outputs[step-2][i] and outputs[step-1][i] > outputs[step-0][i]):
                    # If first ever seen peak, just record it
                    if (lastpeak[i] == -1):
                        lastpeak[i] = step-1
                    # Otherwise, record time since last sighting of a peak
                    else:
                        peaklist[i].append(step-1-lastpeak[i])
                        lastpeak[i] = step-1
            step += 1

        # Second condition: That the oscillation *in each trace* be periodic
        periodic = True
        for i in range(nnsize):
            #print("Peak list: ",i," ",len(peaklist[i])," ",peaklist[i])
            if (len(peaklist[i]) > 5):
                periodvar[i] = np.std(peaklist[i])
            else:
                periodvar[i] = 0.0
                periodic = False
            #print("Period ",i," ",periodvar[i])
            if (periodvar[i] > 1.0):
                periodic = False
        # print("Periodic?",periodic)

        if periodic:
            # Third condition: That the sum has activity above a threshold
            summedactivity /= duration
            #print("Summed activity:", summedactivity)
            if summedactivity > 5.0:
                # Record files
                np.save("OscData/data_%d.npy" %
                        (r), outputs[:int(2/stepsize), :])
                # Visualize traces
                for i in range(nnsize):
                    plt.plot(totaltime, outputs)
                plt.plot(totaltime, summed, 'k')
                plt.xlabel("Time")
                plt.ylabel("Output")
                plt.title("Neural activity")
                plt.show()
