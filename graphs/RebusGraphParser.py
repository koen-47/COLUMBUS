import copy
import re

import pandas as pd
import inflect
import networkx as nx

from .RebusGraph import RebusGraph
from .patterns.Pattern import Pattern
from .templates.Template import Template
from util import get_node_attributes, get_edges_from_node

inflect = inflect.engine()


class RebusGraphParser:
    def __init__(self, compound_words_file_path):
        self._compound_words = pd.read_csv(compound_words_file_path)

    def parse_compound(self, compound=None, graph=None, c1=None, c2=None, is_plural=None):
        if compound is not None:
            if compound not in self._compound_words["stim"].tolist():
                return None
            compound_info = self._compound_words[self._compound_words["stim"] == compound]
            c1, c2 = compound_info["c1"].values[0], compound_info["c2"].values[0]
            is_plural = bool(compound_info["isPlural"].values[0])

        patterns_c1 = Pattern.find_all(c1)
        patterns_c2 = Pattern.find_all(c2)

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

    def parse_idiom(self, idiom):
        idiom_words = [word for word in idiom.split() if word not in Pattern.IGNORE]
        idiom = " ".join(idiom_words)

        idiom_edges = [(i+1, i+2) for i in range(len(idiom.split())-1)]

        graph = RebusGraph()
        graph.add_node(1, text=idiom.split()[0])
        for (i, word), edge in zip(enumerate(idiom.split()[1:]), idiom_edges):
            graph.add_node(i+2, text=word)
            graph.add_edge(edge[0], edge[1], rule=None)

        words = nx.get_node_attributes(graph, "text")
        relational_keywords = Pattern.get_all_relational(as_dict=False)
        while len(set(list(words.values())).intersection(set(relational_keywords))) > 0:
            for node_id, word in words.items():
                if word in Pattern.Relational.OUTSIDE:
                    graph.update_graph_with_rule(node_id, rule="OUTSIDE")
                    break
                elif word in Pattern.Relational.INSIDE:
                    graph.update_graph_with_rule(node_id, rule="INSIDE")
                    break
                elif word in Pattern.Relational.ABOVE:
                    graph.update_graph_with_rule(node_id, rule="ABOVE")
                    break
            words = nx.get_node_attributes(graph, "text")

        # print(graph)
        paths = self.filter_paths(graph)
        # print(paths)
        for path in paths:
            graph = graph.merge_nodes(path)
        graph = nx.convert_node_labels_to_integers(graph, first_label=1)
        graph.graph["template"] = self._select_template(graph)
        for node in graph.nodes:
            graph.nodes[node]["is_plural"] = False

        return graph


    def _select_template(self, graph):
        node_attrs = get_node_attributes(graph)
        if len(node_attrs) == 1:
            node_attrs = node_attrs[1]
            # Check structural rules
            if "highlight" in node_attrs and (node_attrs["highlight"] == "begin" or node_attrs["highlight"] == "after"):
                return {"name": Template.BASE_VERTICAL.name, "obj": Template.BASE_VERTICAL}
            if "size" in node_attrs and node_attrs["size"] == "big":
                return {"name": Template.BASE_VERTICAL.name, "obj": Template.BASE_VERTICAL}
            if "position" in node_attrs and node_attrs["position"] == "high":
                return {"name": Template.SingleNode.HIGH.name, "obj": Template.SingleNode.HIGH}
            if "position" in node_attrs and node_attrs["position"] == "right":
                return {"name": Template.SingleNode.RIGHT.name, "obj": Template.SingleNode.RIGHT}
            if "position" in node_attrs and node_attrs["position"] == "left":
                return {"name": Template.SingleNode.LEFT.name, "obj": Template.SingleNode.LEFT}
            if "position" in node_attrs and node_attrs["position"] == "low":
                return {"name": Template.SingleNode.LOW.name, "obj": Template.SingleNode.LOW}
            if "repeat" in node_attrs and node_attrs["repeat"] == 4:
                return {"name": Template.SingleNode.REPETITION_FOUR.name, "obj": Template.SingleNode.REPETITION_FOUR}
            return {"name": Template.BASE_HORIZONTAL.name, "obj": Template.BASE_HORIZONTAL}
        return {"name": Template.BASE_HORIZONTAL.name, "obj": Template.BASE_HORIZONTAL}

    def filter_paths(self, G):
        filtered_paths = []
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    paths = nx.all_simple_paths(G, source, target)
                    for path in paths:
                        # print(G.e)
                        if all(G.edges[u, v, k].get("rule") is None for u, v, k in
                               zip(path, path[1:], [0] * (len(path)-1))):
                            if not any(set(path) < set(existing_path) for existing_path in filtered_paths):
                                filtered_paths.append(path)

        def remove_sublists(list_of_lists):
            result = []
            for sublist in list_of_lists:
                is_superlist = True
                for other_sublist in list_of_lists:
                    if sublist != other_sublist and set(sublist).issubset(set(other_sublist)):
                        is_superlist = False
                        break
                if is_superlist:
                    result.append(sublist)
            return result

        return remove_sublists(filtered_paths)