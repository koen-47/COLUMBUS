import networkx as nx
import matplotlib.pyplot as plt

from util import get_node_attributes, get_edges_from_node


class RebusGraph(nx.MultiDiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    def add_node(self, node_for_adding, **attr):
        if "text" not in attr:
            raise ValueError("Node must have attribute: text")
        super().add_node(node_for_adding, **attr)

    def update_graph_with_rule(self, node_id, rule):
        in_edges, out_edges = get_edges_from_node(self, node_id)
        prev_node_id = list(in_edges.keys())[0][0]
        next_node_id = list(out_edges.keys())[0][1]
        self.remove_node(node_id)
        self.add_edge(prev_node_id, next_node_id, rule=rule)
        for in_edge, rule in {**in_edges, **out_edges}.items():
            if {"rule": rule} not in list(self[prev_node_id][next_node_id].values()):
                self.add_edge(prev_node_id, next_node_id, rule=rule)

    def visualize(self, attr_offset=0.):
        pos = nx.nx_agraph.graphviz_layout(self)
        node_colors = ["#63D198" for _ in range(len(self.nodes))]
        node_sizes = [2400 for _ in range(len(self.nodes))]
        edge_labels = nx.get_edge_attributes(self, "rule")
        edge_labels = {edge: rule for edge, rule in edge_labels.items() if rule is not None}
        nx.draw(self, pos, with_labels=False, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_size=10)

        node_attrs = get_node_attributes(self)
        for node, attrs in node_attrs.items():
            x, y = pos[node]
            text = attrs["text"]
            del attrs["text"]
            attrs = f"\n{str(attrs)}" if len(attrs) > 0 else ""
            plt.text(x, y + attr_offset, f"{text}{attrs}", fontsize=10, ha="center", va="center")

        plt.margins(0.3)
        plt.show()

    def __str__(self):
        final_str = f"Graph: {self.graph}\n"
        node_attrs = get_node_attributes(self)
        for node, attrs in node_attrs.items():
            final_str += f"Node {node}: {str(attrs)}\n"
        for edge, rule in nx.get_edge_attributes(self, "rule").items():
            final_str += f"Node {edge[0]} -{'-' if rule is None else '- ' + rule + ' --'}> Node {edge[1]}\n"
        return final_str
