import copy

import inflect

from graphs.patterns.Rule import Rule
from graphs.RebusGraph import RebusGraph
from graphs.templates.Template import Template
from util import get_node_attributes

inflect = inflect.engine()


class CompoundRebusGraphParser:
    def __init__(self):
        pass

    def parse(self, c1, c2, is_plural):
        patterns_c1 = Rule.find_all(c1, is_plural)
        patterns_c2 = Rule.find_all(c2, is_plural)

        graph = RebusGraph()
        if len(patterns_c1) > 1 and len(patterns_c2) > 1:
            graph_1, graph_2 = copy.deepcopy(graph), copy.deepcopy(graph)
            text_1 = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph_1.add_node(1, text=text_1, **patterns_c1)
            graph_2.add_node(1, text=c1.upper(), **patterns_c2)
            graph_1.graph["template"] = self._select_template(graph_1)
            graph_2.graph["template"] = self._select_template(graph_2)
            return [graph_1, graph_2]

        if len(patterns_c1) > 1:
            text = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph.add_node(1, text=text, **patterns_c1)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        if len(patterns_c2) > 1:
            graph.add_node(1, text=c1.upper(), **patterns_c2)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        return None

    def _select_template(self, graph):
        n_nodes = len(graph.nodes)
        if n_nodes == 1:
            return {"name": Template.BASE.name, "obj": Template.BASE}
        if n_nodes == 2:
            return {"name": Template.BASE_TWO.name, "obj": Template.BASE_TWO}
        return {"name": Template.BASE_THREE.name, "obj": Template.BASE_THREE}

# def parse_compound(self, compound=None, graph=None, c1=None, c2=None, is_plural=None):
#     if compound is not None:
#         if compound not in self._compound_words["stim"].tolist():
#             return None
#         compound_info = self._compound_words[self._compound_words["stim"] == compound]
#         c1, c2 = compound_info["c1"].values[0], compound_info["c2"].values[0]
#         is_plural = bool(compound_info["isPlural"].values[0])

