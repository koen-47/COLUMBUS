import json

import networkx as nx

from graphs.patterns.Rule import Rule


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

