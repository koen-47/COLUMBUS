import networkx as nx
import inflect

from graphs.RebusGraphParser import RebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter
from graphs.templates import Template

rebus_parser = RebusGraphParser("./saved/ladec_raw_small.csv")
graph = rebus_parser.parse_compound("redcoats")[0]
print(graph)

image_converter = RebusImageConverter()
image_converter.convert_graph_to_image(graph, show=True)
