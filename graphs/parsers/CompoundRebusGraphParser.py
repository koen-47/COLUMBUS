import copy
import itertools
import json
import os

import inflect

from graphs.patterns.Rule import Rule
from graphs.RebusGraph import RebusGraph
from graphs.templates.Template import Template
from util import remove_duplicate_graphs

inflect = inflect.engine()


class CompoundRebusGraphParser:
    def __init__(self):
        # Load homophones
        with open(f"{os.path.dirname(__file__)}/../../saved/homophones_v2.json", "r") as file:
            self._homophones = json.load(file)

    def parse(self, c1, c2, is_plural):
        # Check for patterns for either constituent word
        patterns_c1, conflicts_c1 = Rule.find_all(c1, is_plural)
        patterns_c2, conflicts_c2 = Rule.find_all(c2, is_plural)

        # Format patterns such that all mutually exclusive rules are handled individually
        if len(conflicts_c1) > 1:
            patterns_c1 = [{key: value for key, value in patterns_c1.items() if key not in list([conflict_.split("_")[0] for conflict_ in set(conflicts_c1) - {conflict}])} for conflict in conflicts_c1]
        if len(conflicts_c2) > 1:
            patterns_c2 = [{key: value for key, value in patterns_c2.items() if key not in list([conflict_.split("_")[0] for conflict_ in set(conflicts_c2) - {conflict}])} for conflict in conflicts_c2]

        # List to hold all possible generated graphs
        graphs = []

        # Generate puzzles by combining both words into one
        for pattern_c1 in [patterns_c1] if type(patterns_c1) is not list else patterns_c1:
            graphs += self._generate_rebus(c2, pattern_c1, is_plural)
        for pattern_c2 in [patterns_c2] if type(patterns_c2) is not list else patterns_c2:
            graphs += self._generate_rebus(c1, pattern_c2, is_plural)

        # Generate puzzles by placing both words next to each other
        graphs += self._generate_rebus(c1, {}, is_plural, c2)

        # Remove duplicate graphs
        graphs = remove_duplicate_graphs(graphs)

        return graphs

    def _generate_rebus(self, word_1, rules, is_plural, word_2=None):
        # Create RebusGraph object to store the rebus puzzle representation
        graph = RebusGraph()

        # Only generate the graph if there is more than 1 rule (repeat rule is always there)
        if len(rules) > 1:
            text = self._parse_text(word_1, is_plural)
            graph.add_node(1, text=text, **rules)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        # Generate graph by putting the two constituent words next to each other
        if word_2 is not None:
            word_1_homophone = self._parse_homophones(word_1)
            word_2_homophone = self._parse_homophones(word_2)
            if (word_1 is None and word_2 is None) or (word_1 == word_1_homophone and word_2 == word_2_homophone):
                return []
            graph.add_node(1, text=word_1 if word_1 == word_1_homophone else word_1_homophone)
            graph.add_node(2, text=word_2 if word_2 == word_2_homophone else word_2_homophone)
            graph.add_edge(1, 2, rule="NEXT-TO")
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        # Return empty list if there is no more than 1 rule
        return []

    def _parse_text(self, text, is_plural):
        # Check for homophones
        text = self._parse_homophones(text)

        # Change text in case of plurality
        singular_text = inflect.singular_noun(text)
        if is_plural and singular_text is not False:
            return singular_text.upper()
        return text.upper()

    def _parse_homophones(self, text):
        # Replace text with alternative that is phonetically identical
        if text in self._homophones:
            text = self._homophones[text][0]
            return text
        return text

    def _select_template(self, graph):
        n_nodes = len(graph.nodes)
        if n_nodes == 1:
            return {"name": Template.BASE.name, "obj": Template.BASE}
        if n_nodes == 2:
            return {"name": Template.BASE_TWO.name, "obj": Template.BASE_TWO}
        return {"name": Template.BASE_THREE.name, "obj": Template.BASE_THREE}


