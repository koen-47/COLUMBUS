import copy
import re

import pandas as pd
import inflect
import networkx as nx

from graphs.legacy.RebusGraph import RebusGraph
from graphs.patterns.Rule import Rule
from graphs.templates.Template import Template
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

        patterns_c1, _ = Rule.find_all(c1, is_plural)
        patterns_c2, _ = Rule.find_all(c2, is_plural)

        if graph is None:
            graph = RebusGraph()
        new_node_id = len(graph.nodes) + 1

        if len(patterns_c1) > 1 and len(patterns_c2) > 1:
            graph_1, graph_2 = copy.deepcopy(graph), copy.deepcopy(graph)
            text_1 = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph_1.add_node(new_node_id, text=text_1, **patterns_c1)
            graph_2.add_node(new_node_id, text=c1.upper(), **patterns_c2)
            graph_1.graph["template"] = self._select_template(graph_1)
            graph_2.graph["template"] = self._select_template(graph_2)
            return [graph_1, graph_2]

        if len(patterns_c1) > 1:
            text = inflect.singular_noun(c2).upper() if is_plural else c2.upper()
            graph.add_node(new_node_id, text=text, **patterns_c1)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        if len(patterns_c2) > 1:
            graph.add_node(new_node_id, text=c1.upper(), **patterns_c2)
            graph.graph["template"] = self._select_template(graph)
            return [graph]

        return None

    def parse_idiom(self, idiom):
        idiom_words = [word for word in idiom.split() if word not in Rule.IGNORE]
        idiom = " ".join(idiom_words)

        idiom_edges = [(i+1, i+2) for i in range(len(idiom.split())-1)]

        graph = RebusGraph()
        graph.add_node(1, text=idiom.split()[0])
        for (i, word), edge in zip(enumerate(idiom.split()[1:]), idiom_edges):
            graph.add_node(i+2, text=word)
            graph.add_edge(edge[0], edge[1], rule=None)

        node_attrs = list(get_node_attributes(graph).items())
        skip = False
        for i in range(len(node_attrs)-1):
            if skip:
                skip = False
                continue
            attrs_1, attrs_2 = node_attrs[i][1], node_attrs[i+1][1]
            node = self.parse_compound(c1=attrs_1["text"], c2=attrs_2["text"], is_plural=False)
            if node is not None:
                node = node[0]
                graph.remove_node(i+1)
                graph.remove_node(i+2)
                graph.add_node(i+1, **get_node_attributes(node)[1])
                if i == 1:
                    graph.add_edge(i, i + 1, rule="NEXT-TO")
                elif i > 0:
                    graph.add_edge(i-1, i+1, rule="NEXT-TO")
                if len(graph.nodes) > 1 and i+3 in graph:
                    graph.add_edge(i+1, i+3, rule="NEXT-TO")
                skip = True

        graph = nx.convert_node_labels_to_integers(graph, first_label=1)
        words = nx.get_node_attributes(graph, "text")
        relational_keywords = Rule.get_all_relational(as_dict=False)
        while len(set(list(words.values())).intersection(set(relational_keywords))) > 0:
            for node_id, word in words.items():
                if word in Rule.Relational.OUTSIDE:
                    graph.update_graph_with_rule(node_id, rule="OUTSIDE")
                    break
                elif word in Rule.Relational.INSIDE:
                    graph.update_graph_with_rule(node_id, rule="INSIDE")
                    break
                elif word in Rule.Relational.ABOVE:
                    graph.update_graph_with_rule(node_id, rule="ABOVE")
                    break
            words = nx.get_node_attributes(graph, "text")

        paths = self.filter_paths(graph)
        for path in paths:
            graph = graph.merge_nodes(path)
        graph = nx.convert_node_labels_to_integers(graph, first_label=1)
        graph.graph["template"] = self._select_template(graph)

        for node in graph.nodes:
            if "repeat" not in graph.nodes[node]:
                graph.nodes[node]["repeat"] = 1
            graph.nodes[node]["text"] = graph.nodes[node]["text"].upper()

        mapping = {node_2: node_1 for node_1, node_2 in zip(range(1, len(graph.nodes)+1), nx.topological_sort(graph))}
        graph = nx.relabel_nodes(graph, mapping)

        return graph

    def _select_template(self, graph):
        node_attrs = get_node_attributes(graph)
        inside_count = sum(1 for u, v, attrs in graph.edges(data=True) if
                           all(attrs.get(attr) == value for attr, value in {"rule": "INSIDE"}.items()))
        n_nodes = len(node_attrs) - inside_count

        if n_nodes == 1:
            node_attrs = node_attrs[1]
            if "highlight" in node_attrs and (node_attrs["highlight"] == "begin" or node_attrs["highlight"] == "after"):
                return {"name": Template.BASE.name, "obj": Template.BASE}
            if "size" in node_attrs and node_attrs["size"] == "big":
                return {"name": Template.BASE.name, "obj": Template.BASE}
            return {"name": Template.BASE.name, "obj": Template.BASE}
        if n_nodes == 2:
            return {"name": Template.BASE_TWO.name, "obj": Template.BASE_TWO}
        return {"name": Template.BASE_THREE.name, "obj": Template.BASE_THREE}

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