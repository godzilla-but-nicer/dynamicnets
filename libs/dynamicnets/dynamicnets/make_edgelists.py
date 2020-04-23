from network_filtering import generate_networks
import networkx as nx
import os
from tqdm import tqdm

MAX_NODES = 3

for n in range(2, MAX_NODES + 1):
    print('Writing edgelists for', n, 'nodes')
    G_list = generate_networks(n)

    # Check if directory for storing these lists exists
    dir_name = 'data/' + str(n) + '_node_edgelists/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # write edgelists to file
    for i, G in tqdm(enumerate(G_list)):
        file_name = dir_name + str(n) + '_node_' + str(i) + '.edgelist' 
        fout = open(file_name, 'wb')
        nx.write_edgelist(G, fout)
        fout.close()