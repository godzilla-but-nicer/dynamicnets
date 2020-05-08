from tqdm import tqdm
from time import time
from scipy.special import comb
from itertools import combinations

## This code was lightly adapted from Zackary Dunnivan

def shapley_full(df,model,endog,exog,fout):
    # n is the size of the largest sets of permutations
    n = len(exog)
    n_combs = 0 
    n_combs = sum([comb(16,k) for k in range(16)])

    #final_matrix = np.zeros((n, n_combs/n))
    final_matrix = [[] for x in range(n)]

    def get_rsquared_for_sets(sets):
        for s in sets:
            features = [exog[i] for i in s]
            fs = []
            for f in features:
                if '+' in f:
                    for ef in f.split('+'):
                        fs.append(ef)
                else:
                    fs.append(f)
            features = fs
            this_model = model(data=df,formula="%s ~ %s" % (endog[0],'+'.join(features)))
            results = this_model.fit(maxiter=5000,disp=False)
            # for OLS
            rsquared = results.rsquared_adj
            # for poisson
            # rsquared = pearsonr(df[endog[0]],this_model.predict(results.params))[0]
            yield (s,rsquared)

    def concat_tuple(tup,final):
        state=()  
        for i in tup:  
            state=state+(i,)
        state = state + (final,)
        return state

    start_time = time()

    # these are our R2 for single variable models
    rsquareds = dict()
    for combo,rsquared in get_rsquared_for_sets([(i,) for i in range(n)]):
        rsquareds[str(combo[-1])] = rsquared
        adjusted_value = (comb(n-1,0))**-1*(rsquared-0) # the prior rsquared is 0, for the model with no dependent variables
        final_matrix[combo[-1]].append(adjusted_value)

    for k in tqdm(range(2,n+1)):
        for combo,rsquared in get_rsquared_for_sets(combinations(range(n),k-1)): 
            combo_string = '.'.join(map(str,sorted(combo)))
            rsquareds[combo_string] = rsquared

    combo,rsquared = list(get_rsquared_for_sets([tuple(range(n))]))[0]
    rsquareds['.'.join(map(str,sorted(combo)))] = rsquared
                      
    # calculate the difference for adding in the new variable
    for k in range(2,n+1):
        for i in range(n):
            all_but_i = list(range(n))
            del all_but_i[i]
            for prior_combo in combinations(all_but_i,k-1):
                combo = '.'.join(map(str,sorted(list(prior_combo)+[i])))
                prior_combo = '.'.join(map(str,sorted(prior_combo)))
                diff = rsquareds[combo] - rsquareds[prior_combo]
                final_matrix[i].append((comb(n-1,k-1))**-1*diff)
      
    print("Ran for %d minutes." % int((time()-start_time)/60))

    # final model
    combo,rsquared = list(get_rsquared_for_sets([tuple(range(n))]))[0]
        
    phis = [1/n*sum(final_matrix[i]) for i in range(n)]
    fout.write('rsquared: ' + str(rsquared) + '\n')
    fout.write('shapely_computed: ' + str(sum(phis)) + '\n')
    for i in range(n):
        #print("%s: %s, %i" % (exog[i],' '.join(map(str,final_matrix[i])), np.mean(final_matrix[i])))
        fout.write("%s: %.4f, %.2f%%\n" % (exog[i], phis[i], (phis[i]/rsquared)*100))