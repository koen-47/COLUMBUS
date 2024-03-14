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


def strike_through(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result