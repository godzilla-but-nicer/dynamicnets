import numpy as np
import glob
import pickle
import pandas as pd
from MGA import Microbial
from ThresholdBooleanNetwork import ThresholdBooleanNetwork
from WeightMatrixStorage import WeightMatrix, WeightMatrixStorage

# datafile where network structure is stored
edgelist_dir = snakemake.input[0]
edgelist_list = glob.glob(edgelist_dir + '*')
nnsize = int(edgelist_list[0].split('/')[-1].split('_')[0])
print('nnsize', nnsize)

# NN params
weight_range = 15

# timescale params
fit_duration = 10

# different things occur on different time scales
fittime = np.arange(0.0, fit_duration)  # what does this mean?


# The sum of absolute differences in each trace and their past
def fitnessFunction(genotype):
    global edgelist  # import the current edgelist
    # initialize network and set paramss in some bounds
    nn = ThresholdBooleanNetwork(edgelist)
    nn.ga_set_weights(genotype, weight_range)

    # calculate fitness over the fitness timescale
    fit = 0.0
    for t in fittime:
        past_states = nn.states
        nn.synchronous_step()
        new_states = nn.states
        fit += np.sum(abs(new_states - past_states))
    return fit/(nnsize * fit_duration)


# EA params
popsize = 25
genesize = nnsize * nnsize
recombProb = 0.5
mutatProb = 0.1
generations = 50
tournaments = generations * popsize
fitness_threshold = 12.0
strength_threshold = 5.0
trials = 10

# we're going to save a pkl object of weight matrices
# as well as a dataframe of summary stats
df_list = []
wms = WeightMatrixStorage()

for edgelist in edgelist_list:
    print('Evolving network based on', edgelist)
    struct_id = edgelist.split('/')[-1].split('_')[2].split('.')[0]  # id filenames
    
    # get summary statistics
    avg_fit = []
    best_fit = []

    for r in range(trials):
        print('\tTrial', r)
        # Evolve and visualize fitness over generations
        ga = Microbial(fitnessFunction, popsize,
                       genesize, recombProb, mutatProb)
        ga.run(tournaments)

        # Get best evolved network and show its activity
        afit, bfit, bi = ga.fitStats()

        # init ctrnn so we can save weight matrix
        nn = ThresholdBooleanNetwork(edgelist)
        nn.ga_set_weights(bi, weight_range)

        # create weight matrix object
        wm = WeightMatrix(nnsize, nn.weight_matrix, edgelist, bfit)
        wms.add_matrix(wm)

        # add per run values to our lists
        avg_fit.append(afit)
        best_fit.append(bfit)
    
    # outside of the evolution trial loop turn these lists into statistics
    avg_avg_fit = np.mean(avg_fit)
    min_best_fit = min(best_fit)
    max_best_fit = max(best_fit)
    avg_best_fit = np.mean(best_fit)

    row = {'nodes': nnsize,
           'id': struct_id,
           'avg_avg_fit': avg_avg_fit,
           'min_best_fit': min_best_fit,
           'max_best_fit': max_best_fit,
           'avg_best_fit': avg_best_fit}
    df_list.append(row)

# first the dataframe
df = pd.DataFrame(df_list)
dfout = open(snakemake.output.csv, 'w')
df.to_csv(dfout)
dfout.close()

# second is all of the weight matrices
wm_file = open(snakemake.output.wm, 'wb')
pickle.dump(wms, wm_file)
wm_file.close()