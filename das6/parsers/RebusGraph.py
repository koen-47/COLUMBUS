import networkx as nx
import matplotlib.pyplot as plt

from util import get_node_attributes, get_edges_from_node
from parsers.patterns.Rule import Rule


class RebusGraph(nx.DiGraph):
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
        node_attrs = get_node_attributes(self).copy()
        for node, attrs in node_attrs.items():
            if "icon" in attrs:
                attrs["text"] = list(attrs["icon"].values())[0]
            attr_str = [f"{rule}: {value}" for rule, value in attrs.items() if rule != "icon" and rule != "sound"]
            if "icon" in attrs:
                attr_str += [f"icon: ({list(attrs['icon'].keys())[0]}: {list(attrs['icon'].values())[0]})"]
            if "sound" in attrs:
                sound_rule = list(attrs['sound'].keys())[0]
                sound_value = list(attrs['sound'].values())[0]
                if isinstance(sound_value, list):
                    sound_value = sound_value[0]
                attr_str += [f"sound: ({sound_rule}: {sound_value})"]
            attr_str = ", ".join(attr_str)
            final_str += f"Node {node} attributes: ({attr_str})\n"
        for i, (edge, rule) in enumerate(nx.get_edge_attributes(self, "rule").items()):
            final_str += f"Edge {i+1}: node {edge[0]} to node {edge[1]} (rule: {rule})\n"
        return final_str
