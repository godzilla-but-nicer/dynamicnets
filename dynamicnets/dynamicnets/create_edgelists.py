from NetworkSampler import NetworkSampler

nb = NetworkSampler(min_nodes=min(snakemake.wildcards.nodes),
                    sample_sizes=500)
nb.write_edgelists()
