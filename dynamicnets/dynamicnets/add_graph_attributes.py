import pandas as pd
import networkx as nx
import glob

# assign data paths to variables
edgelist_dir = snakemake.input[0]
evo_df_file = snakemake.input[1]
edgelists = glob.glob(edgelist_dir + '*')
nodes = int(edgelists[0].split('/')[-1].split('_')[0])

new_rows = []
# load edgelists
for edgelist in edgelists:
    print('\tnew graph')
    # get number of nodes and edgelist id
    edgelist_id = edgelist.split('/')[-1].split('_')[2].split('.')[0]
    G = nx.read_edgelist(edgelist, create_using=nx.DiGraph())

    # count the number of edges
    num_edges = len(list(G.edges()))
        
    # count cycles, long cycles are more than 2 nodes
    cycles = nx.simple_cycles(G)
    long_cycles = 0
    two_cycles = 0
    for cycle in cycles:
        if len(cycle) > 2:
            long_cycles += 1
        else:
            two_cycles += 1
        
    is_strongly_connected = nx.is_strongly_connected(G)

    new_row = {'nodes': nodes,
               'id': edgelist_id,
               'num_edges': num_edges,
               'two_cycles': two_cycles,
               'long_cycles': long_cycles,
               'strongly_connected': is_strongly_connected}
    new_rows.append(new_row)

# load old dataframe and add new rows, theres some error so we need to
# coerce the types to be ints for the columns we merge on
evo_df = pd.read_csv(evo_df_file)
evo_df['nodes'] = evo_df['nodes'].astype(int)
evo_df['id'] = evo_df['id'].astype(int)
new_df = pd.DataFrame(new_rows)
new_df['nodes'] = new_df['nodes'].astype(int)
new_df['id'] = new_df['id'].astype(int)
df = evo_df.merge(new_df, on=['nodes', 'id'])
df.to_csv('data/evo_graph_stats/{}_node_graph.csv'.format(nodes))
