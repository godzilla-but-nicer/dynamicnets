# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:21:21 2019

@author: Lauren Benson
"""

from MGA import Microbial
from CTRNNEdgelist import CTRNN
import numpy as np
import pickle
import glob
import pandas as pd


# datafile where network structure is stored
edgelist_dir = snakemake.input[0]
edgelist_list = glob.glob(edgelist_dir + '*')
nnsize = int(edgelist_list[0].split('/')[-1].split('_')[0])

# NN params
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
time = np.arange(0.0, duration, stepsize)  # where we care about preformance
transientdur = np.arange(0.0, transient, stepsize)  # before we care
totaltime = np.arange(0.0, duration+transient, stepsize)


# The sum of absolute differences in each trace and their past
def fitnessFunction(genotype):
    global edgelist  # import the current edgelist
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
popsize = 25
genesize = nnsize*nnsize + 2*nnsize
recombProb = 0.5
mutatProb = 0.1
generations = 50
tournaments = generations * popsize
fitness_threshold = 20.0
strength_threshold = 5.0
trials = 10

# we're going to save a pkl object of passing timeseries
# as well as a dataframe of summary stats
passing_timeseries = {}
df_list = []

for edgelist in edgelist_list:
    print('Evolving network based on', edgelist)
    struct_id = edgelist.split('/')[-1].split('_')[2].split('.')[0]  # id filenames
    passing_timeseries.update({struct_id: []})

    # we're going to save a bunch of summary statistics
    # hopefully more than we need
    avg_fit = []
    best_fit = []
    good_enough = []
    peak_sd = []
    strong_together = []

    # Repeat evolution to find circuits where:
    #   1. Oscillations are strongly enough
    #   2. Each individual trace is periodic
    #   3. activities sum above a threshold
    for r in range(trials):
        print('\tTrial', r)
        # Evolve and visualize fitness over generations
        ga = Microbial(fitnessFunction, popsize,
                       genesize, recombProb, mutatProb)
        ga.run(tournaments)

        # Get best evolved network and show its activity
        afit, bfit, bi = ga.fitStats()

        # these can be made true in conditionals but we always want them
        # to start false
        first_condition = False
        third_condition = False

        # First condition: That the oscillation be strong
        if bfit > fitness_threshold:
            first_condition = True
            print('\t\tFitness > Threshold!')

            # init ctrnn
            nn = CTRNN(edgelist)
            nn.setParameters(bi, WeightRange, BiasRange,
                             TimeConstMin, TimeConstMax)
            nn.initializeState(np.zeros(nnsize))

            # init arrays for downstream calculations
            outputs = np.zeros((len(totaltime), nnsize))
            summed = np.zeros(len(totaltime))
            lastpeak = np.zeros(nnsize) - 1  # fill w/ -1
            periodvar = np.zeros(nnsize)
            # init empty lists for nodes
            peaklist = [[] for _ in range(nnsize)]

            # iterate over the transient timescale and aggregate network output
            # by taking the mean
            step = 0
            for t in transientdur:
                nn.step(stepsize)
                outputs[step] = nn.Output
                summed[step] = np.mean(nn.Output)
                step += 1

            # iterate over the full timescale and calculate links activity
            # except all of outrest of the full timescale (transient + time)
            # step is incremented below the nested loop
            summedactivity = 0.0
            for t in time:
                summedPastOutputs = np.sum(nn.Output)
                nn.step(stepsize)
                summedCurrentOutputs = np.sum(nn.Output)
                # why do we care about the cumulative absolute difference
                # between two consecutive timesteps?
                summedactivity += np.sum(abs(summedCurrentOutputs -
                                             summedPastOutputs)/stepsize)
                outputs[step] = nn.Output
                summed[step] = np.mean(nn.Output)

                # Find period for each neuron
                for i in range(nnsize):
                    if (outputs[step-1][i] > outputs[step-2][i] and
                            outputs[step-1][i] > outputs[step-0][i]):
                        # If first ever seen peak, just record it
                        if (lastpeak[i] == -1):
                            lastpeak[i] = step - 1
                        # Otherwise, record time since last sighting of a peak
                        else:
                            peaklist[i].append(step-1-lastpeak[i])
                            lastpeak[i] = step - 1
                step += 1

            # Second condition: oscillation *in each trace* be periodic
            periodic = True
            for i in range(nnsize):
                if (len(peaklist[i]) > 5):  # if we have a bunch of peaks
                    periodvar[i] = np.std(peaklist[i])
                else:  # if we don't have a bunch of peaks
                    periodvar[i] = 0.0
                    periodic = False
                if (periodvar[i] > 1.0):  # if std dev too big
                    periodic = False

            if periodic:
                # Third condition: That the sum has activity above a threshold
                summedactivity /= duration
                if summedactivity > strength_threshold:
                    third_condition = True
                    # add to list of timeseries to save
                    passing_timeseries[struct_id].append(outputs)
        
        # add per run values to our lists
        avg_fit.append(afit)
        best_fit.append(bfit)
        good_enough.append(first_condition)
        peak_sd.append(np.mean(periodvar))
        strong_together.append(strong_together)

    # outside of the evolution trial loop turn these lists into statistics
    avg_avg_fit = np.mean(avg_fit)
    min_best_fit = min(best_fit)
    max_best_fit = max(best_fit)
    avg_best_fit = np.mean(best_fit)
    prop_good_enough = np.mean(good_enough)

    avg_peak_sd = np.mean(peak_sd)  # remember that this is over subset of runs
    prop_together = np.mean(third_condition)

    # combine these into dictionary that will become df row
    row = {'nodes': nnsize,
           'id': struct_id,
           'avg_avg_fit': avg_avg_fit,
           'min_best_fit': min_best_fit,
           'max_best_fit': max_best_fit,
           'avg_best_fit': avg_best_fit,
           'prop_fit_enough': prop_good_enough,
           'avg_peak_sd': avg_peak_sd,
           'prop_together': prop_together}
    print('dict done')

    df_list.append(row)
    print('dict added to list')

# once all of our looping is done we can save our data
# first is the time series of runs that passed all three checks
tsout = open(snakemake.output.pkl, 'wb')
pickle.dump(passing_timeseries, tsout)
tsout.close()

# second the dataframe
df = pd.DataFrame(df_list)
dfout = open(snakemake.output.csv, 'w')
df.to_csv(dfout)
dfout.close()
