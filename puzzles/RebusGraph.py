import networkx as nx

from util import get_node_attributes


class RebusGraph(nx.DiGraph):
    """
    Class that contains information on a rebus graph (inherits from Networkx's DiGraph class).
    """
    def __init__(self, **attr):
        super().__init__(**attr)

    def add_node(self, node_for_adding, **attr):
        """
        Add a new node to the graph (must contain the 'text' attribute).

        :param node_for_adding: ID of new node.
        :param attr: attributes of new node.
        """
        if "text" not in attr:
            raise ValueError("Node must have attribute: text")
        super().add_node(node_for_adding, **attr)

    def compute_difficulty(self, adjust_for_size=True):
        """
        Computes the difficulty of each graph. The difficulty is defined as the average number of rules per node (that
        are not basic rules, such as the text associated with a node or when it is repeated once) + the number of edges
        that do not have a NEXT-TO relation.

        :param adjust_for_size: flag to denote if the difficulty to adjusted based on the size of the graph.
        :return: pair containing number of individual rules and number of edge rules (as defined by the description
        above)
        """
        n_ind_rules = 0
        n_rel_rules = 0
        node_attrs = get_node_attributes(self)
        for node, attrs in node_attrs.items():
            attrs_ = attrs.copy()
            del attrs_["text"]
            if attrs_["repeat"] == 1:
                del attrs_["repeat"]
            n_ind_rules += len(attrs_)
        for edge, rule in nx.get_edge_attributes(self, "rule").items():
            if rule != "NEXT-TO":
                n_rel_rules += 1
        if adjust_for_size:
            return n_ind_rules / len(node_attrs), n_rel_rules
        return n_ind_rules, n_rel_rules

    def __str__(self):
        final_str = f"Graph: {self.graph}\n"
        node_attrs = get_node_attributes(self)
        for node, attrs in node_attrs.items():
            final_str += f"Node {node}: {str(attrs)}\n"
        for edge, rule in nx.get_edge_attributes(self, "rule").items():
            final_str += f"Node {edge[0]} -{'-' if rule is None else '-(' + rule + ')-'}> Node {edge[1]}\n"
        return final_str
