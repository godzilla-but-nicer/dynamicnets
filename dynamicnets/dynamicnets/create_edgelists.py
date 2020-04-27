from NetworkSampler import NetworkSampler

nb = NetworkSampler(min_nodes=min(snakemake.config["net_sizes"].keys()),
                    max_nodes=max(snakemake.config["net_sizes"].keys()),
                    sample_sizes=500)
nb.write_edgelists()
