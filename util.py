import networkx as nx

from graphs.patterns.Pattern import Pattern


def get_node_attributes(graph):
    attrs = {attr: nx.get_node_attributes(graph, attr) for attr in ["text", "is_plural"] + Pattern.ALL_RULES}
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
