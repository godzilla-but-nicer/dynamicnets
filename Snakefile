from os.path import join as j

configfile: "workflow/config.yaml"

DATA_DIR = config["data_dir"]

PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

rule all:
    input:
        PAPER, SUPP

rule paper:
    input:
        PAPER_SRC, SUPP_SRC
    params:
        paper_dir = PAPER_DIR
    output:
        PAPER, SUPP
    shell:
        "cd {params.paper_dir}; make"

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
