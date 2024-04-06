import pandas as pd

from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter
from graphs.patterns.Rule import Rule
import util

compounds = pd.read_csv("./saved/ladec_raw.csv")

parser = CompoundRebusGraphParser()
generator = RebusImageConverter()
graphs = parser.parse("left", "feet", True)

print(f"Number of graphs generated: {len(graphs)}")
for graph in graphs:
    generator.convert_graph_to_image(graph, show=True)
