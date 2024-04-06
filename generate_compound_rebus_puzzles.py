import pandas as pd
from tqdm import tqdm

from graphs.parsers.RebusGraphParser import RebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter

compounds = pd.read_csv("./saved/ladec_raw_small.csv")
compounds_common = pd.read_csv("./saved/ladec_common_raw_small.csv")

rebus_parser = RebusGraphParser("./saved/ladec_raw_small.csv")
rebus_image = RebusImageConverter()

n_puzzles_all, n_puzzles_common = 0, 0
for compound in tqdm(compounds["stim"], desc="Generating rebus puzzles (all)"):
    try:
        graphs = rebus_parser.parse_compound(compound)
        if graphs is not None:
            if len(graphs) == 1:
                rebus_image.convert_graph_to_image(graphs[0], save_path=f"./results/compounds/all/{compound}.png")
            else:
                for i, graph in enumerate(graphs):
                    rebus_image.convert_graph_to_image(graph, save_path=f"./results/compounds/all/{compound}_{i+1}.png")
            n_puzzles_all += len(graphs)
    except AttributeError:
        pass

for compound in tqdm(compounds_common["stim"], desc="Generating rebus puzzles (common)"):
    try:
        graphs = rebus_parser.parse_compound(compound)
        if graphs is not None:
            if len(graphs) == 1:
                rebus_image.convert_graph_to_image(graphs[0], save_path=f"./results/compounds/common/{compound}.png")
            else:
                for i, graph in enumerate(graphs):
                    rebus_image.convert_graph_to_image(graph, save_path=f"./results/compounds/common/{compound}_{i+1}.png")
            n_puzzles_common += len(graphs)
    except AttributeError:
        pass

print(f"Number of rebus puzzles (all): {n_puzzles_all}")
print(f"Number of rebus puzzles (common): {n_puzzles_common}")
