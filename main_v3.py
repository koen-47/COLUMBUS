import pandas as pd

from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter
from graphs.patterns.Rule import Rule
import util

# parser = PhraseRebusGraphParser()
# graph = parser.parse("punch above one's weight outside of the box")
# graph = parser.parse("go back in time")
# graph = parser.parse("cross my heart and hope to die")

compounds = pd.read_csv("./saved/ladec_raw.csv")

parser = CompoundRebusGraphParser()
generator = RebusImageConverter()
graphs = parser.parse("fore", "front", False)

print(f"Number of graphs generated: {len(graphs)}")
for graph in graphs:
    print(graph)
    generator.convert_graph_to_image(graph, show=True)
