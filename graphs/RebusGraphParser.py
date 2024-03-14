import copy

import pandas as pd
import inflect

from .RebusGraph import RebusGraph
from .patterns.Pattern import Pattern

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

        print(patterns_c1)
        print(patterns_c2)

        if graph is None:
            graph = RebusGraph()
        new_node_id = len(graph.nodes) + 1

        if len(patterns_c1) > 1 and len(patterns_c2) > 1:
            graph_1, graph_2 = copy.deepcopy(graph), copy.deepcopy(graph)
            graph_1.graph["template"] = patterns_c1["template"]
            graph_2.graph["template"] = patterns_c2["template"]
            text_1 = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph_1.add_node(new_node_id, text=text_1, is_plural=is_plural, **patterns_c1)
            graph_2.add_node(new_node_id, text=c1.upper(), is_plural=False, **patterns_c2)
            return [graph_1, graph_2]

        if len(patterns_c1) > 1 or patterns_c1["template"] != "base":
            graph.graph["template"] = patterns_c1["template"]
            text = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph.add_node(new_node_id, text=text, is_plural=is_plural, **patterns_c1)
            return [graph]

        if len(patterns_c2) > 1 or patterns_c2["template"] != "base":
            graph.graph["template"] = patterns_c2["template"]
            graph.add_node(new_node_id, text=c1.upper(), is_plural=False, **patterns_c2)
            return [graph]

        return None
