import pandas as pd
import statsmodels.formula.api as smf
import os
import glob
from shapley_decomposition import shapley_full

# load dataframes
dir_path = snakemake.input[0]

df_files = glob.glob(dir_path + '*')

df_list = [pd.read_csv(csv, index_col=0) for csv in df_files]

# combine them
df = pd.concat(df_list)

# make a separate dataframe with columns normalized by nodes
norm_cols = ['num_edges', 'two_cycles', 'long_cycles']
df_norm = df
for col in norm_cols:
    df_norm[col] = df[col] / df['nodes']

if os.path.exists(snakemake.output[0]):
    os.remove(snakemake.output[0])

fout = open(snakemake.output[0], 'a')

# perform all regressions and decompositions
response_cols = ['avg_distance']
df_list = [df, df_norm]
for i in range(2):
    if i == 0:
        fout.write('Original Model:\n')
    else:
        fout.write('Normalized Model\n')

    # for each response variable run the regression and decomp
    for response in response_cols:
        if i == 0:
            indep = ['nodes', 'num_edges', 'two_cycles', 'long_cycles',
                     'long_cycles', 'strongly_connected']
            regress_string = response + ' ~ nodes + num_edges + two_cycles + long_cycles + strongly_connected'
        else:
            indep = ['two_cycles', 'num_edges', 'long_cycles',
                     'strongly_connected']
            regress_string = response + ' ~ num_edges + two_cycles + long_cycles + strongly_connected'
        # run regression and write out the results
        result = smf.ols(regress_string, data=df_list[i]).fit()
        fout.write(str(result.summary()))
        fout.write('\n\np-values:\n')

        # write out the pvalues as well
        for p in result.pvalues:
            fout.write(str(p) + '\n')
        fout.write('Shapley:\n\n')

        # decomposition
        shapley_full(df_list[i], smf.ols, [response], indep, fout)
        fout.write('\n\n\n')

fout.close()