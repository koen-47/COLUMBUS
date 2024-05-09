import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from util import get_node_attributes, get_edge_information, get_graph_as_sequence
from .templates.Template import Template


class RebusImageConverterV2:
    def __init__(self):
        self.BASE_SIZE = (400, 400)

    def generate(self, graph, show=False):
        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))
        node_attrs = get_node_attributes(graph)
        template = self._select_template(graph)

        for element, (node, attrs) in zip(template.elements, node_attrs.items()):
            x, y, _ = element
            points = self._apply_repetition_rule(attrs, x, y, _)
            for point in points:
                x, y, _ = point
                text = attrs["text"]
                text = self._apply_direction_rule(text, attrs)
                ax.text(x, y, text, fontsize=36, fontweight="bold", fontfamily="Consolas", ha="center", va="center")
                self._apply_highlight_rule(attrs, ax, text, x, y, _)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if show:
            plt.show()

    def generate_inside(self, graph, show=False):
        graph_sequence = get_graph_as_sequence(graph)
        inside_left = graph_sequence[:graph_sequence.index("INSIDE")]
        print(inside_left)
        inside_right = graph_sequence[graph_sequence.index("INSIDE")+1:]
        print(inside_right)

        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))
        template = graph.graph["template"]["obj"]
        (x, y), size = template.elements[0][:2], template.elements[0][2] * 40
        text = ax.text(x, y, text_inside, color="black", ha="center", va="center", weight="bold",
                       fontsize=size, fontfamily="Consolas")
        ax.annotate(text_outside_left, xycoords=text, xy=(0, 0), va="bottom", ha="right", color="black", weight="bold",
                    fontsize=size,
                    fontfamily="Consolas")
        ax.annotate(text_outside_right, xycoords=text, xy=(1, 0), va="bottom", color="black", weight="bold",
                    fontsize=size,
                    fontfamily="Consolas")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if show:
            plt.show()
        plt.close(fig)

    def _select_template(self, graph):
        n_nodes = len(graph.nodes)
        if n_nodes == 1:
            return Template.BASE
        if n_nodes == 2:
            return Template.BASE_TWO
        if n_nodes == 3:
            return Template.BASE_THREE

    def _apply_repetition_rule(self, attrs, x, y, size):
        n_repeats = 1 if "repeat" not in attrs else attrs["repeat"]
        if n_repeats == 1:
            return [(x, y, size)]
        sizes = [40 * (1 / n_repeats)] * n_repeats

        # ACROSS X-DIMENSION
        if "direction" in attrs and (attrs["direction"] == "down" or attrs["direction"] == "up"):
            return list(zip(np.linspace(x - 0.15, x + 0.15, n_repeats).tolist(), [y] * n_repeats, sizes))

        y_gap = n_repeats * 0.05
        # ACROSS Y-DIMENSION
        return list(zip([x] * n_repeats, np.linspace(y - y_gap, y + y_gap, n_repeats).tolist(), sizes))


    def _apply_direction_rule(self, text, attrs):
        if "direction" in attrs:
            if attrs["direction"] == "reverse":
                return text[::-1]
            if attrs["direction"] == "down":
                return "\n".join(text)
            if attrs["direction"] == "up":
                return "\n".join(text)[::-1]
        return text

    def _apply_highlight_rule(self, attrs, ax, text, x, y, size):
        if "highlight" in attrs:
            if attrs["highlight"] == "after":
                x += (0.11 * (len(text) / 2))
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.025)
                ab = AnnotationBbox(imagebox, (x, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.025)
                ab = AnnotationBbox(imagebox, (x, y - 0.12), frameon=False)
                ax.add_artist(ab)
            if attrs["highlight"] == "before":
                x -= (0.11 * (len(text) / 2))
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.025)
                ab = AnnotationBbox(imagebox, (x, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.025)
                ab = AnnotationBbox(imagebox, (x, y - 0.12), frameon=False)
                ax.add_artist(ab)
            if attrs["highlight"] == "middle":
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.025)
                ab = AnnotationBbox(imagebox, (x, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.025)
                ab = AnnotationBbox(imagebox, (x, y - 0.12), frameon=False)
                ax.add_artist(ab)
