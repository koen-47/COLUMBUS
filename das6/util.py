import glob
import json
import os

import networkx as nx
import pandas as pd

from parsers.patterns.Rule import Rule


def get_node_attributes(graph):
    attrs = {attr: nx.get_node_attributes(graph, attr) for attr in ["text", "is_plural"] + Rule.ALL_RULES}
    node_attrs = {}
    for attr, nodes in attrs.items():
        for node, attr_val in nodes.items():
            if node not in node_attrs:
                node_attrs[node] = {attr: attr_val}
            node_attrs[node][attr] = attr_val
    return node_attrs


def get_edges_from_node(graph, node_id):
    in_edges, out_edges = {}, {}
    for edge in graph.in_edges(node_id, keys=True):
        in_edges[edge] = nx.get_edge_attributes(graph, "rule")[edge]
    for edge in graph.out_edges(node_id, keys=True):
        out_edges[edge] = nx.get_edge_attributes(graph, "rule")[edge]
    return [in_edges, out_edges]


def get_edge_information(graph):
    node_attrs = get_node_attributes(graph)
    edge_attrs = nx.get_edge_attributes(graph, "rule")
    edge_info = {}
    for edge in graph.edges:
        rule = edge_attrs[edge]
        source = node_attrs[edge[0]]
        target = node_attrs[edge[1]]
        edge_info[edge] = (source, rule, target)
    return edge_info


def remove_duplicate_graphs(graphs):
    unique_graphs = []
    for graph in graphs:
        is_duplicate = False
        for unique_graph in unique_graphs:
            if nx.utils.graphs_equal(graph, unique_graph):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_graphs.append(graph)
    return unique_graphs


def count_relational_rules(phrase):
    relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
    return sum([1 for word in phrase.split() if word in relational_keywords])


def get_answer_graph_pairs():
    from parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
    from parsers.PhraseRebusGraphParser import PhraseRebusGraphParser

    phrases = [os.path.basename(file).split(".")[0]
               for file in glob.glob(f"{os.path.dirname(__file__)}/data/images/*")]
    ladec = pd.read_csv(f"{os.path.dirname(__file__)}/data/misc/ladec_raw_small.csv")
    custom_compounds = pd.read_csv(f"{os.path.dirname(__file__)}/data/misc/custom_compounds.csv")

    compound_parser = CompoundRebusGraphParser()
    phrase_parser = PhraseRebusGraphParser()
    phrase_to_graph = {}
    compound_to_graph = {}
    for phrase in phrases:
        parts = phrase.split("_")
        index = 0
        # print(parts, ladec["stim"])
        if (parts[0] in ladec["stim"].tolist() and len(parts) == 2) or len(parts) == 1:
            if parts[-1].isnumeric():
                index = int(parts[-1]) - 1
                parts = parts[:-1]
            phrase_ = " ".join(parts)
            row = ladec.loc[ladec["stim"] == phrase_].values.flatten().tolist()
            if len(row) == 0:
                row = custom_compounds.loc[custom_compounds["stim"] == phrase_].values.flatten().tolist()
            c1, c2, is_plural = row[0], row[1], bool(row[2])
            graphs = compound_parser.parse(c1, c2, is_plural)
            compound_to_graph[phrase] = graphs[index]
        else:
            if parts[-1].isnumeric():
                index = int(parts[-1]) - 1
                parts = parts[:-1]
            phrase_ = " ".join(parts)
            graphs = phrase_parser.parse(phrase_)
            phrase_to_graph[phrase] = graphs[index]

    return phrase_to_graph, compound_to_graph
