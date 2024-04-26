import json

import pandas as pd

from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.RebusImageConverter import RebusImageConverter
from graphs.patterns.Rule import Rule
from util import count_relational_rules
from graphs.RebusImageConverterV2 import RebusImageConverterV2

parser = PhraseRebusGraphParser()
generator = RebusImageConverterV2()

# graph = parser.parse("punch one's hippo lion tiger above weight")
# graph = parser.parse("cross roads")
# graph = parser.parse("wind back the clock")
# graph = parser.parse("cross my heart and hope to die")
# graph = parser.parse("cross my heart and hope to die chilling cold")
# graph = parser.parse("in the mood")
# graph = parser.parse("in and through the mood")
# graph = parser.parse("through out of the mood")
# graph = parser.parse("once the car was inside")
# graph = parser.parse("car up in the air")
# graph = parser.parse("for good and all")
# graph = parser.parse("up a height")

graph = parser.parse("clean up one's act")
print(graph[0])

# generator.generate(graph[0])

# graph = parser.parse("stars in one's eyes")
# generator.generate(graph)

# graph = parser.parse("once in a blue moon")
# generator.generate(graph)

# compounds = pd.read_csv("./saved/ladec_raw.csv")

# parser = CompoundRebusGraphParser()
# generator = RebusImageConverter()
# graphs = parser.parse("hope", "to", True)

# print(f"Number of graphs generated: {len(graphs)}")
# for graph in graphs:
#     print(graph)
#     generator.convert_graph_to_image(graph, show=True)

# relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
# with open("./saved/idioms_raw.json", "r") as file:
#     idioms = json.load(file)
#     counter = 0
#     for idiom in idioms:
#         if count_relational_rules(idiom) <= 1 and idiom.split()[0] not in relational_keywords and idiom.split()[-1] not in relational_keywords:
#             counter += 1
#             print(idiom)
#     print(counter)

# relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
# with open("./saved/idioms_raw.json", "r") as file:
#     idioms = json.load(file)
#     counter = 0
#     for idiom in idioms:
#         if count_relational_rules(idiom) > 1:
#             counter += 1
#             # print(idiom)
#     print(counter)
#     print(counter/len(idioms))
#     print((counter/len(idioms))*100)

# individual_keywords = [x for xs in Rule.get_all_rules()["individual"].values() for x in xs]
# with open("./saved/idioms_raw.json", "r") as file:
#     idioms = json.load(file)
#     counter = 0
#     for idiom in idioms:
#         words = idiom.split()
#         for i in range(len(words)):
#             pass
#
#     print(counter)
