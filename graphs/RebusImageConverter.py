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
        graph_attrs = graph.graph
        if graph_attrs["template"] == Template.BASE.name:
            self._convert_base_template(ax, graph=graph)
        elif graph_attrs["template"] == Template.HIGH.name:
            self._convert_repetition_template(ax, graph=graph, template=Template.HIGH)
        elif graph_attrs["template"] == Template.REPETITION_FOUR.name:
            self._convert_repetition_template(ax, graph=graph, template=Template.REPETITION_FOUR)
        elif graph_attrs["template"] == Template.REPETITION_TWO.name:
            self._convert_repetition_template(ax, graph=graph, template=Template.REPETITION_TWO)

        if save_path != "":
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def _convert_base_template(self, ax, graph):
        node_attrs = get_node_attributes(graph)
        elements = Template.BASE.elements if not graph.graph["is_plural"] else Template.BASE.plural_elements
        for element in elements:
            for node, attrs in node_attrs.items():
                x, y = element
                size = 40 * Template.BASE.size
                text = self._apply_reverse_rule(attrs)
                color = self._apply_color_rule(attrs)
                ax.text(x, y, text, fontsize=40, fontweight="bold", fontfamily="Consolas", color=color, ha="center",
                        va="center")
                self._apply_cross_rule(attrs, ax, text, x, y, size)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _convert_repetition_template(self, ax, graph, template):
        node_attrs = get_node_attributes(graph)
        elements = template.elements if not graph.graph["is_plural"] else template.plural_elements
        for element in elements:
            for node, attrs in node_attrs.items():
                x, y = element
                size = 40 * Template.REPETITION_FOUR.size
                text = self._apply_reverse_rule(attrs)
                color = self._apply_color_rule(attrs)
                ax.text(x, y, text, fontsize=size, fontweight="bold", fontfamily="Consolas", color=color, ha="center",
                        va="center")
                self._apply_cross_rule(attrs, ax, text, x, y, size)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

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
