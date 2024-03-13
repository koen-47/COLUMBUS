import networkx as nx
import matplotlib.pyplot as plt

from .patterns.Pattern import Pattern
from util import get_node_attributes


class RebusGraph(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    def add_node(self, node_for_adding, **attr):
        if "text" not in attr:
            raise ValueError("Node must have attribute: text")
        super().add_node(node_for_adding, **attr)

    def visualize(self, attr_offset=0.):
        pos = nx.nx_agraph.graphviz_layout(self)
        node_colors = ["#63D198" for _ in range(len(self.nodes))]
        node_sizes = [2400 for _ in range(len(self.nodes))]
        nx.draw(self, pos, with_labels=False, node_color=node_colors, node_size=node_sizes)

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
            final_str += f"Node {node}: {str(attrs)}"
        return final_str
