import networkx as nx
import matplotlib.pyplot as plt

from util import get_node_attributes, get_edges_from_node
from graphs.patterns.Rule import Rule


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
            if {"rule": rule} not in list(self[prev_node_id][next_node_id].values()) and rule is not None:
                self.add_edge(prev_node_id, next_node_id, rule=rule)

    def merge_nodes(self, path):
        new_node = path[0]
        concatenated_text = " ".join(self.nodes[n]["text"] for n in path)
        combined_attrs = {k: [] for k in self.nodes[path[0]].keys()}
        for u, v in zip(path[:-1], path[1:]):
            self = nx.contracted_nodes(self, path[0], v, self_loops=False)
            for key in combined_attrs:
                if key != "text":
                    combined_attrs[key].append(self.nodes[u][key])

        self = nx.relabel_nodes(self, {path[0]: new_node})
        for key in combined_attrs:
            if key != "text":
                self.nodes[new_node][key] = ''.join(combined_attrs[key])
        self.nodes[new_node]["text"] = concatenated_text
        return self

    def visualize(self, attr_offset=0.):
        pos = nx.nx_agraph.graphviz_layout(self)
        node_colors = ["#63D198" for _ in range(len(self.nodes))]
        node_sizes = [3600 for _ in range(len(self.nodes))]
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

    def compute_difficulty(self, adjust_for_size=True):
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
