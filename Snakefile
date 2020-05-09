from os.path import join as j

configfile: "workflow/config.yaml"

DATA_DIR = config["data_dir"]

rule all:
    input:
        

rule generate_edgelists:
    input:
        "data/"
    output:
        "data/edgelists/{nodes}_node_edgelists"
    script:
        "dynamicnets/dynamicnets/create_edgelists.py"

rule evolve_networks:
    input:
        "data/edgelists/{nodes}_node_edgelists/"
    output:
        pkl="data/passing_timeseries/{nodes}_nodes_passing_timeseries.pkl",
        csv="data/evo_stats/{nodes}_node_evolve.csv",
        wm="data/weight_matrices/{nodes}_nodes_weight_matrices.pkl"
    log:
        "logs/evolve_networks_{nodes}.log"
    script:
        "dynamicnets/dynamicnets/evolve_CTRNN_osc.py"

rule add_graph_attributes:
    input:
        "data/edgelists/{nodes}_node_edgelists/",
        "data/evo_stats/{nodes}_node_evolve.csv"
    output:
        "data/evo_graph_stats/{nodes}_node_graph.csv"
    script:
        "dynamicnets/dynamicnets/add_graph_attributes.py"

rule evolve_bnn:
    input:
        "data/edgelists/{nodes}_node_edgelists/"
    output:
        csv="data/evo_stats_bnn/{nodes}_node_bnn_evolve.csv",
        wm="data/weight_matrices_bnn/{nodes}_nodes_bnn_weight_matrices.pkl"
    script:
        "dynamicnets/dynamicnets/evolve_BNN_osc.py"

rule add_graph_attributes_bnn:
    input:
        "data/edgelists/{nodes}_node_edgelists/",
        "data/evo_stats_bnn/{nodes}_node_bnn_evolve.csv"
    output:
        "data/evo_graph_stats_bnn/{nodes}_node_graph_bnn.csv"
    script:
        "dynamicnets/dynamicnets/add_graph_attributes.py"

rule add_timeseries_attributes_bnn:
    input:
        edges="data/edgelists/{nodes}_node_edgelists/",
        csv="data/evo_graph_stats_bnn/{nodes}_node_graph_bnn.csv",
        mat="data/weight_matrices_bnn/{nodes}_nodes_bnn_weight_matrices.pkl"
    output:
        "data/evo_timeseries_bnn/{nodes}_node_timeseries_bnn.csv"
    script:
        "dynamicnets/dynamicnets/add_timeseries_attributes.py"

rule fitness_regression:
    input:
        "data/evo_graph_stats/"
    output:
        "data/fitness_regressions/regress_output.txt"
    script:
        "dynamicnets/dynamicnets/run_fitness_regressions.py"

rule fitness_regression_bnn:
    input:
        "data/evo_timeseries_bnn/"
    output:
        "data/fitness_regressions/bnn_regress_output.txt"
    script:
        "dynamicnets/dynamicnets/run_fitness_regressions_bnn.py"

rule diversity_regression_bnn:
    input:
        "data/evo_timeseries_bnn/"
    output:
        "data/diversity_regressions/bnn_diversity.txt"
    script:
        "dynamicnets/dynamicnets/run_diversity_regressions.py"

rule norm_amp_regressions_bnn:
    input:
        "data/evo_timeseries_bnn/"
    output:
        "data/amplitude_regression/bnn_amplitude.txt"
    script:
        "dynamicnets/dynamicnets/run_amp_regressions.py"
