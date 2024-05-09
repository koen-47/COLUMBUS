import itertools
import json
import os
import copy

from ..RebusGraph import RebusGraph
from ..patterns.Rule import Rule
from .CompoundRebusGraphParser import CompoundRebusGraphParser

class PhraseRebusGraphParser:
    def __init__(self):
        with open(f"{os.path.dirname(__file__)}/../../saved/ignore_words.json", "r") as file:
            self._ignore_words = json.load(file)

    def _is_valid(self, phrase):
        if phrase == "":
            return False
        relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
        phrase_parts = phrase.split()
        n_rel_keywords = [word for word in phrase_parts if word in relational_keywords]
        if len(n_rel_keywords) > 1:
            return False
        if phrase_parts[0].lower() in relational_keywords:
            return False
        if phrase_parts[-1].lower() in relational_keywords:
            return False
        return True

    def parse(self, phrase):
        phrase_words = [word for word in phrase.split() if word not in self._ignore_words]
        phrase = " ".join(phrase_words)
        if not self._is_valid(phrase):
            return None

        graphs_per_word = self._get_all_graphs_per_word(phrase)
        combinations = list(itertools.product(*graphs_per_word))

        all_graphs = []
        for c in combinations:
            relational_nodes = []
            graph = list(c)[0].copy()
            n_nodes = len(graph.nodes)
            for sub_graph in list(c)[1:]:
                if isinstance(sub_graph, RebusGraph):
                    n_nodes += len(sub_graph.nodes)
                    for node in sub_graph.nodes(data=True):
                        graph.add_node(len(graph.nodes)+1, **node[1])
                        graph.add_edge(len(graph.nodes)-1, len(graph.nodes), rule="NEXT-TO")
                else:
                    relational_nodes.append((sub_graph, n_nodes, n_nodes+1))
            all_graphs.append(graph)
            for edge in relational_nodes:
                graph[edge[1]][edge[2]]["rule"] = edge[0]

        return all_graphs

    def _get_all_graphs_per_word(self, phrase):
        compound_parser = CompoundRebusGraphParser()
        divided_words = self._divide_text(phrase)

        graphs_per_word = []
        skip = False
        for words in divided_words:
            for rule, keywords in Rule.get_all_rules()["relational"].items():
                if words[0] in keywords:
                    graphs_per_word.append([rule.upper()])
                    skip = True
                    break
            if skip:
                skip = False
                continue
            i = 0
            while i < len(words) - 1:
                graphs = compound_parser.parse(c1=words[i], c2=words[i + 1], is_plural=False)
                if len(graphs) > 0:
                    graphs_per_word.append(graphs)
                    words.pop(i)
                    words.pop(i)
                    i -= 1
                else:
                    graph = RebusGraph()
                    graph.add_node(1, text=words[i].upper(), repeat=1)
                    graphs_per_word.append([graph])
                i += 1
            if i < len(words):
                graph = RebusGraph()
                graph.add_node(1, text=words[i].upper(), repeat=1)
                graphs_per_word.append([graph])

        return graphs_per_word

    def _divide_text(self, phrase):
        relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
        phrase_words = phrase.split()
        divided_words = []

        i = 0
        while i < len(phrase_words):
            word = phrase_words[i]
            if word in relational_keywords:
                divided_words.append(phrase_words[:i])
                divided_words.append([phrase_words[i]])
                phrase_words = phrase_words[i+1:]
                i = -1
            i += 1
        divided_words += [phrase_words]
        return divided_words

