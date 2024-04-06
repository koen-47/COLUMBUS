import copy
import itertools
import json
import os

import inflect

from graphs.patterns.Rule import Rule
from graphs.RebusGraph import RebusGraph
from graphs.templates.Template import Template

inflect = inflect.engine()


class CompoundRebusGraphParser:
    def __init__(self):
        pass

    def parse(self, c1, c2, is_plural):
        # Check for patterns for either constituent word
        patterns_c1, conflicts_c1 = Rule.find_all(c1, is_plural)
        patterns_c2, conflicts_c2 = Rule.find_all(c2, is_plural)

        # Format patterns such that all mutually exclusive rules are handled individually
        patterns_c1 = [{key: value for key, value in patterns_c1.items() if key not in list([conflict_.split("_")[0] for conflict_ in set(conflicts_c1) - {conflict}])} for conflict in conflicts_c1]
        patterns_c2 = [{key: value for key, value in patterns_c2.items() if key not in list([conflict_.split("_")[0] for conflict_ in set(conflicts_c2) - {conflict}])} for conflict in conflicts_c2]

        # Generate puzzles
        graphs = []
        for pattern_c1 in patterns_c1:
            graphs += self._generate_rebus("", c2, pattern_c1, {}, is_plural)
        for pattern_c2 in patterns_c2:
            graphs += self._generate_rebus(c1, "", {}, pattern_c2, is_plural)

        return graphs

    def _generate_rebus(self, c1, c2, patterns_c1, patterns_c2, is_plural):
        graph = RebusGraph()
        if len(patterns_c1) > 1 and len(patterns_c2) > 1:
            graph_1 = copy.deepcopy(graph)
            text_1 = self._parse_text(c2, is_plural)
            graph_1.add_node(1, text=text_1, **patterns_c1)
            graph_1.graph["template"] = self._select_template(graph_1)

            graph_2 = copy.deepcopy(graph)
            text_2 = self._parse_text(c1, is_plural)
            graph_2.add_node(1, text=text_2, **patterns_c2)
            graph_2.graph["template"] = self._select_template(graph_2)
            return [graph_1, graph_2]

        if len(patterns_c1) > 1:
            text = self._parse_text(c2, is_plural)
            graph.add_node(1, text=text, **patterns_c1)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        if len(patterns_c2) > 1:
            text = self._parse_text(c1, is_plural)
            graph.add_node(1, text=text, **patterns_c2)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        return []

    def _parse_text(self, text, is_plural):
        with open(f"{os.path.dirname(__file__)}/../../saved/homophones_v2.json", "r") as file:
            homophones = json.load(file)

        if text in homophones:
            text = homophones[text][0]
            return text
        singular_text = inflect.singular_noun(text)
        if is_plural and singular_text is not False:
            return singular_text.upper()
        return text.upper()

    def _select_template(self, graph):
        n_nodes = len(graph.nodes)
        if n_nodes == 1:
            return {"name": Template.BASE.name, "obj": Template.BASE}
        if n_nodes == 2:
            return {"name": Template.BASE_TWO.name, "obj": Template.BASE_TWO}
        return {"name": Template.BASE_THREE.name, "obj": Template.BASE_THREE}


