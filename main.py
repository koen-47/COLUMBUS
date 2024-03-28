import json

import networkx as nx
import inflect
import pandas as pd

from graphs.RebusGraphParser import RebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter
from util import get_edge_information
from scraping.WiktionaryIdiomsWebScraper import WiktionaryIdiomsWebScraper

# scraper = WiktionaryIdiomsWebScraper()
# scraper.scrape()

# with open("./saved/theidioms_raw.json", "r") as file:
#     theidioms = [idiom["idiom"] for idiom in json.load(file)]
# with open("./saved/wiktionary_idioms_raw.json", "r") as file:
#     wiktionary = json.load(file)
# with open("./saved/idioms_raw.json", "w") as file:
#     idioms = list(set(theidioms + wiktionary))
#     json.dump(idioms, file)

rebus_parser = RebusGraphParser("./saved/ladec_raw_small.csv")
image_converter = RebusImageConverter()
#
# graph = rebus_parser.parse_idiom("all in time")
# print(graph)
# # image_converter.render_inside_rule_puzzle(graph)
# graph = rebus_parser.parse_idiom("all in all")
# print(graph)
# # image_converter.render_inside_rule_puzzle(graph)



# graph = rebus_parser.parse_compound("foreground")[0]
# graph.visualize()
#
# image_converter.convert_graph_to_image(graph, show=True)

# image_converter.render_inside_rule_puzzle(graph)

# image_converter.convert_graph_to_image(graph, show=True)

# rebus_parser.parse_idiom("out of sight")
# rebus_parser.parse_idiom("bent out of shape")
# rebus_parser.parse_idiom("go out on a limb")
# rebus_parser.parse_idiom("a fish out of water")
# rebus_parser.parse_idiom("left out in the cold")
# rebus_parser.parse_idiom("once in a blue moon")

# rebus_parser.parse_idiom("knight in shining armour")
# rebus_parser.parse_idiom("standing so I do")
# rebus_parser.parse_idiom("when in Rome, do as the Romans")
# rebus_parser.parse_idiom("Rome, do as the Romans")

with open("./saved/theidioms_raw.json", "r") as file:
    idioms = pd.DataFrame(json.load(file))

counter = 0
for idiom in idioms["idiom"]:
    if "in" in idiom.split() and "in" != idiom.split()[0] and "in" != idiom.split()[-1]:
        graph = rebus_parser.parse_idiom(idiom)
        image_converter.render_inside_rule_puzzle(graph)
        counter += 1
print(counter)


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
