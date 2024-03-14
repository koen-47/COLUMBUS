from itertools import cycle

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from util import get_node_attributes
from .templates.Template import Template


class RebusImageConverter:
    def __init__(self):
        self.BASE_SIZE = (400, 400)

    def convert_graph_to_image(self, graph, show=False, save_path=""):
        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))
        template = self._select_template(graph)
        self._convert_template(ax, graph, template)

        if save_path != "":
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def _convert_template(self, ax, graph, template):
        node_attrs = get_node_attributes(graph)
        for element, (node, attrs) in zip(template.elements, cycle(node_attrs.items())):
            if attrs["is_plural"]:
                for plural_element in element["plural"]:
                    self._render_text(ax, plural_element, attrs)
            else:
                self._render_text(ax, element["singular"], attrs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _render_text(self, ax, element, attrs):
        (x, y), size = element[:2], element[2] * 40
        text = self._apply_reverse_rule(attrs)
        color = self._apply_color_rule(attrs)
        ax.text(x, y, text, fontsize=size, fontweight="bold", fontfamily="Consolas", color=color, ha="center",
                va="center")
        self._apply_cross_rule(attrs, ax, text, x, y, size)

    def _apply_color_rule(self, attrs):
        return "black" if "color" not in attrs else attrs["color"]

    def _apply_reverse_rule(self, attrs):
        return attrs["text"] if "reverse" not in attrs else attrs["text"][::-1]

    def _apply_cross_rule(self, attrs, ax, text, x, y, size):
        if "cross" in attrs:
            line_x1, line_x2 = x - (0.0025 * size * (len(text) / 2)), x + (0.0025 * size * (len(text) / 2))
            line = ConnectionPatch((line_x1, y), (line_x2, y), "axes fraction", "axes fraction",
                                   color="black", lw=2)
            ax.add_artist(line)

    def _select_template(self, graph):
        graph_attrs = graph.graph
        if len(graph.nodes) == 1:
            if graph_attrs["template"] == Template.BASE.name:
                return Template.BASE
            elif graph_attrs["template"] == Template.SingleNode.REPETITION_FOUR.name:
                return Template.SingleNode.REPETITION_FOUR
            elif graph_attrs["template"] == Template.SingleNode.HIGH.name:
                return Template.SingleNode.HIGH
            elif graph_attrs["template"] == Template.SingleNode.RIGHT.name:
                return Template.SingleNode.RIGHT
        return None
