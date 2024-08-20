import copy
import glob
import os

import networkx as nx
import pandas as pd

from puzzles.patterns.Rule import Rule


def get_node_attributes(graph):
    """
    Gets the node attributes of the specified graph through a dictionary that maps each node ID to the attributes of
    that node.

    :param graph: graph to get the node attributes for.
    :return: a dictionary mapping each node ID in the specified graph to the attributes of that node.
    """

    attrs = {attr: nx.get_node_attributes(graph, attr) for attr in ["text", "is_plural"] + Rule.ALL_RULES}
    node_attrs = {}
    for attr, nodes in attrs.items():
        for node, attr_val in nodes.items():
            if node not in node_attrs:
                node_attrs[node] = {attr: attr_val}
            node_attrs[node][attr] = attr_val
    return node_attrs


def get_graph_as_sequence(graph):
    """
    Splits a graph into subgraphs based on the relational rules (edges) in it.

    :param graph: graph to split.
    :return: a sequence (list) of the nodes in a graph, separated by the relational rules that are not a NEXT-TO rule.
    """
    node_attrs = get_node_attributes(graph)
    edge_attrs = nx.get_edge_attributes(graph, "rule")
    sequence = []
    for node, attrs in node_attrs.items():
        if (node-1, node) in edge_attrs and edge_attrs[(node-1, node)] != "NEXT-TO":
            sequence.append(edge_attrs[(node-1, node)])
        sequence.append(attrs)
    return sequence


def remove_duplicate_graphs(graphs):
    """
    Removes duplicate graphs from a list of graphs.

    :param graphs: list of graphs.
    :return: deduplicated list of graphs.
    """
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


def get_answer_graph_pairs(combine=False):
    """
    Gets the graphs of the answers of each rebus puzzle.

    :param combine: combine graphs belonging to phrases and compounds into one dictionary.
    :return: a dictionary mapping to answer of a rebus puzzle to its graph.
    """
    from puzzles.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
    from puzzles.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser

    # Load input data
    phrases = [os.path.basename(file).split(".")[0]
               for file in glob.glob(f"{os.path.dirname(__file__)}/results/benchmark/images/*")]
    ladec = pd.read_csv(f"{os.path.dirname(__file__)}/data/input/ladec_raw_small.csv")
    custom_compounds = pd.read_csv(f"{os.path.dirname(__file__)}/data/input/custom_compounds.csv")

    compound_parser = CompoundRebusGraphParser()
    phrase_parser = PhraseRebusGraphParser()
    phrase_to_graph = {}
    compound_to_graph = {}

    for phrase in phrases:
        orig_phrase = phrase
        if phrase.endswith("_icon") or phrase.endswith("_non-icon"):
            phrase = "_".join(phrase.split("_")[:-1])
        parts = phrase.split("_")
        index = 0
        if (parts[0] in ladec["stim"].tolist() and len(parts) == 2) or len(parts) == 1:
            if parts[-1].isnumeric():
                index = int(parts[-1]) - 1
                parts = parts[:-1]
            phrase_ = " ".join(parts)
            row = ladec.loc[ladec["stim"] == phrase_].values.flatten().tolist()
            if len(row) == 0:
                row = custom_compounds.loc[custom_compounds["stim"] == phrase_].values.flatten().tolist()
            c1, c2, is_plural = row[0], row[1], bool(row[3])
            graphs = compound_parser.parse(c1, c2, is_plural)
            phrase = "_".join(orig_phrase.split())
            if orig_phrase.endswith("non-icon"):
                compound_to_graph[phrase] = remove_icons_from_graph(graphs[index])
            else:
                compound_to_graph[phrase] = graphs[index]

        else:
            if parts[-1].isnumeric():
                index = int(parts[-1]) - 1
                parts = parts[:-1]
            phrase_ = " ".join(parts)
            graphs = phrase_parser.parse(phrase_)
            phrase = "_".join(orig_phrase.split())
            if orig_phrase.endswith("non-icon"):
                phrase_to_graph[phrase] = remove_icons_from_graph(graphs[index])
            else:
                phrase_to_graph[phrase] = graphs[index]
    if combine:
        graphs = {}
        graphs.update(phrase_to_graph)
        graphs.update(compound_to_graph)
        return graphs

    return phrase_to_graph, compound_to_graph


def remove_icons_from_graph(graph):
    """
    Converts the icons in a graph back to its textual counterpart.

    :param graph: graph containing an icon.
    :return: graph where all nodes with icon rules have been converted back to their textual counterpart.
    """
    graph_no_icon = copy.deepcopy(graph)
    graph_no_icon_node_attrs = get_node_attributes(graph_no_icon)
    for attr in graph_no_icon_node_attrs.values():
        if "icon" in attr:
            attr["text"] = list(attr["icon"].keys())[0].upper()
            del attr["icon"]

    for node in graph_no_icon.nodes:
        graph_no_icon.nodes[node].clear()
    nx.set_node_attributes(graph_no_icon, graph_no_icon_node_attrs)
    return graph_no_icon
