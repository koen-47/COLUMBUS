import networkx as nx
import inflect

from graphs.RebusGraphParser import RebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter

rebus_parser = RebusGraphParser("./saved/ladec_raw_small.csv")
# graph = rebus_parser.parse_compound(c1="blue", c2="moon", is_plural=False)[0]
# print(graph)

# rebus_parser.parse_idiom("out of sight")
rebus_parser.parse_idiom("bent out of shape")
rebus_parser.parse_idiom("go out on a limb")
rebus_parser.parse_idiom("a fish out of water")
rebus_parser.parse_idiom("left out in the cold")

# image_converter = RebusImageConverter()
# image_converter.convert_graph_to_image(graph, show=True)

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def text_circle(text, radius, fontsize=12, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(4, 4))
#
#     num_chars = len(text)
#     angles = np.linspace(0, 2*np.pi, num_chars, endpoint=False)
#
#     for angle, char in zip(angles, text):
#         x = 0.5 + radius * np.cos(angle)
#         y = 0.5 + radius * np.sin(angle)
#         print(x, y)
#         ax.text(x, y, char, ha='center', va='center', fontsize=fontsize)
#
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.axis('off')
#
#     return ax
#
# # Example usage:
# text = "abcd"
# text_circle(text, radius=0.1, fontsize=20)
# plt.show()
