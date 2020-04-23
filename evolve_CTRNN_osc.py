# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:21:21 2019

@author: Lauren Benson
"""

import mga
import CTRNN_Lauren
import numpy as np
import matplotlib.pyplot as plt

# NN Params
nnsize = 3
fitduration=10
duration = 40
transient = 10
stepsize = 0.05
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0
TimeConstMax = 2.0

#starting_outputs = np.arange(0.1,1.0,0.8) # 2^3 = 8 * 10 * 50 = 4000 total evaluations
#testing_outputs = np.arange(0.1,1.0,0.1) # 9^3 = 729 saved

fittime = np.arange(0.0,fitduration,stepsize)
time = np.arange(0.0,duration,stepsize)
transientdur = np.arange(0.0,transient,stepsize)
totaltime = np.arange(0.0,duration+transient,stepsize)

# Fitness function:
#   The sum/multiplication of absolute differences in each trace and their past
def fitnessFunction(genotype):
    nn = CTRNN_Lauren.CTRNN(nnsize)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    fit = 0.0
    trials = 0
#    for o1 in starting_outputs:
#        for o2 in starting_outputs:
#            for o3 in starting_outputs:
#                nn.initializeOutput(np.array([o1,o2,o3])) # XXX
    nn.initializeState(np.zeros(nnsize))
    for t in fittime:
        pastOutputs = nn.Output
        nn.step(stepsize)
        currentOutputs = nn.Output
        fit += np.sum(abs(currentOutputs - pastOutputs)/stepsize)
    trials += 1
    return fit/(nnsize*fitduration*trials)

# EA Params
popsize = 10
genesize = nnsize*nnsize + 2*nnsize
recombProb = 0.5
mutatProb = 0.1
generations = 50
tournaments = generations * popsize

# Repeat evolution to find circuits that:
#   1. Oscillate strongly enough
#   2. Where each trace is periodic
#   3. Where the sum has activity above a threshold (neuron output does not cancel out)
for r in range(10):
    print(r)
    # Evolve and visualize fitness over generations
    ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb)
    ga.run(tournaments)
    ga.showFitness()
    # Get best evolved network and show its activity 
    af,bf,bi = ga.fitStats()
    #print("Fitness:",bf)
    # First condition: That the oscillation be strong
    
    if bf > 5.0:        # XXX Can that five be turned into something dependent on paramers?
        #print("Running simulation")
        nn = CTRNN_Lauren.CTRNN(nnsize)
        nn.setParameters(bi,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#        for o1 in testing_outputs:
#            for o2 in testing_outputs:
#                for o3 in testing_outputs:
                    #nn.initializeOutput(np.array([o1,o2,o3])) # XXX
        nn.initializeState(np.zeros(nnsize))
        outputs = np.zeros((len(totaltime),nnsize))
        summed = np.zeros(len(totaltime))
        lastpeak=np.zeros(nnsize)-1
        periodvar=np.zeros(nnsize)
        peaklist = [[],[],[]]  # XXX Dependent on number of neurons?
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
            summedactivity += np.sum(abs(summedCurrentOutputs - summedPastOutputs)/stepsize)
            outputs[step] = nn.Output
            summed[step] = np.mean(nn.Output)
            ## Find period for each neuron
            for i in range(nnsize):
                if (outputs[step-1][i]>outputs[step-2][i] and outputs[step-1][i]>outputs[step-0][i]):
                    ## If first ever seen peak, just record it
                    if (lastpeak[i]==-1):
                        lastpeak[i] = step-1
                    ## Otherwise, record time since last sighting of a peak
                    else:
                        peaklist[i].append(step-1-lastpeak[i])
                        lastpeak[i] = step-1
            step += 1
        
        # Second condition: That the oscillation *in each trace* be periodic
        periodic = True
        for i in range(nnsize):
            #print("Peak list: ",i," ",len(peaklist[i])," ",peaklist[i])
            if (len(peaklist[i])>5):
                periodvar[i] = np.std(peaklist[i])
            else:
                periodvar[i] = 0.0
                periodic = False
            #print("Period ",i," ",periodvar[i])
            if (periodvar[i]>1.0):
                periodic = False
        #print("Periodic?",periodic)   
        
        if periodic:
            # Third condition: That the sum has activity above a threshold 
            summedactivity /= duration
            #print("Summed activity:", summedactivity)
            if summedactivity > 5.0:
                # Record files 
                np.save("OscData/data_%d.npy"%(r),outputs[:int(2/stepsize),:])
                # Visualize traces
                for i in range(nnsize):
                    plt.plot(totaltime,outputs)
                plt.plot(totaltime,summed,'k')
                plt.xlabel("Time")
                plt.ylabel("Output")
                plt.title("Neural activity")
                plt.show()