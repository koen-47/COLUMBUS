import copy

import pandas as pd
import inflect

from .RebusGraph import RebusGraph
from .patterns.Pattern import Pattern
from .templates.Template import Template
from util import get_node_attributes

inflect = inflect.engine()


class RebusGraphParser:
    def __init__(self, compound_words_file_path):
        self._compound_words = pd.read_csv(compound_words_file_path)

    def parse_compound(self, compound, graph=None):
        if compound not in self._compound_words["stim"].tolist():
            return None
        compound_info = self._compound_words[self._compound_words["stim"] == compound]
        c1, c2 = compound_info["c1"].values[0], compound_info["c2"].values[0]
        is_plural = bool(compound_info["isPlural"].values[0])

        patterns_c1 = Pattern.find_all(c1)
        patterns_c2 = Pattern.find_all(c2)

        # print(patterns_c1)
        # print(patterns_c2)

        if graph is None:
            graph = RebusGraph()
        new_node_id = len(graph.nodes) + 1

        if len(patterns_c1) > 0 and len(patterns_c2) > 0:
            graph_1, graph_2 = copy.deepcopy(graph), copy.deepcopy(graph)
            text_1 = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph_1.add_node(new_node_id, text=text_1, is_plural=is_plural, **patterns_c1)
            graph_2.add_node(new_node_id, text=c1.upper(), is_plural=is_plural, **patterns_c2)
            graph_1.graph["template"] = self._select_template(graph_1)
            graph_2.graph["template"] = self._select_template(graph_2)
            return [graph_1, graph_2]

        if len(patterns_c1) > 0:
            text = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph.add_node(new_node_id, text=text, is_plural=is_plural, **patterns_c1)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        if len(patterns_c2) > 0:
            graph.add_node(new_node_id, text=c1.upper(), is_plural=is_plural, **patterns_c2)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        return None

    def _select_template(self, graph):
        node_attrs = get_node_attributes(graph)
        if len(node_attrs) == 1:
            node_attrs = node_attrs[1]
            # Check structural rules
            if "position" in node_attrs and node_attrs["position"] == "high":
                return {"name": Template.SingleNode.HIGH.name, "obj": Template.SingleNode.HIGH}
            if "position" in node_attrs and node_attrs["position"] == "right":
                return {"name": Template.SingleNode.RIGHT.name, "obj": Template.SingleNode.RIGHT}
            if "repeat" in node_attrs and node_attrs["repeat"] == 4:
                return {"name": Template.SingleNode.REPETITION_FOUR.name, "obj": Template.SingleNode.REPETITION_FOUR}
            return {"name": Template.BASE.name, "obj": Template.BASE}
