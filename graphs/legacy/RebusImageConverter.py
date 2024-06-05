from itertools import cycle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import ConnectionPatch

from util import get_node_attributes, get_edges_from_node, get_edge_information
from graphs.templates.Template import Template


class RebusImageConverter:
    def __init__(self):
        self.BASE_SIZE = (400, 400)

    def convert_graph_to_image(self, graph, show=False, save_path=""):
        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))
        self._convert_template(ax, graph, template=graph.graph["template"]["obj"])
        if save_path != "":
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def _convert_template(self, ax, graph, template):
        node_attrs = dict(sorted(get_node_attributes(graph).items()))
        for element, (node, attrs) in zip(template.elements, node_attrs.items()):
            self._render_text(ax, element, attrs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _render_text(self, ax, element, attrs):
        (x, y), size = element[:2], element[2] * 40
        points = self._apply_repetition_rule(attrs, x, y, size)
        for point in points:
            # x, y, size = 0.5, point, 40
            x, y, size = point
            text = attrs["text"]
            size *= (1/len(text)) * 4
            text = self._apply_direction_rule(text, attrs)
            color = self._apply_color_rule(attrs)
            size = self._apply_size_rule(size, attrs)
            offset, alignment = self._apply_position_rule(attrs)
            x += offset[0]
            y += offset[1]
            ax.text(x, y, text, fontsize=size, fontweight="bold", fontfamily="Consolas", color=color, ha=alignment[0],
                    va=alignment[1])
            self._apply_cross_rule(attrs, ax, text, x, y, size)
            self._apply_highlight_rule(attrs, ax, text, x, y, size)

    def _apply_repetition_rule(self, attrs, x, y, size):
        n_repeats = 1 if "repeat" not in attrs else attrs["repeat"]
        if n_repeats == 1:
            return [(x, y, size)]
        sizes = [40 * (1/n_repeats)] * n_repeats

        # ACROSS X-DIMENSION
        if "direction" in attrs and (attrs["direction"] == "down" or attrs["direction"] == "up"):
            return list(zip(np.linspace(x - 0.15, x + 0.15, n_repeats).tolist(), [y] * n_repeats, sizes))
        return list(zip([x] * n_repeats, np.linspace(y - 0.2, y + 0.2, n_repeats).tolist(), sizes))

    def _apply_color_rule(self, attrs):
        return "black" if "color" not in attrs else attrs["color"]

    def _apply_cross_rule(self, attrs, ax, text, x, y, size):
        if "cross" in attrs:
            line_x1, line_x2 = x - (0.0025 * size * (len(text) / 2)), x + (0.0025 * size * (len(text) / 2))
            line = ConnectionPatch((line_x1, y), (line_x2, y), "axes fraction", "axes fraction",
                                   color="black", lw=2)
            ax.add_artist(line)

    def _apply_position_rule(self, attrs):
        if "position" in attrs:
            if attrs["position"] == "high":
                return (0., 0.4), ("center", "center")
            if attrs["position"] == "low":
                return (0., -0.4), ("center", "center")
            if attrs["position"] == "left":
                return (-0.4, 0), ("left", "center")
            if attrs["position"] == "right":
                return (0.4, 0), ("right", "center")
        return (0, 0), ("center", "center")

    def _apply_direction_rule(self, text, attrs):
        if "direction" in attrs:
            if attrs["direction"] == "reverse":
                return text[::-1]
            if attrs["direction"] == "down":
                return "\n".join(text)
            if attrs["direction"] == "up":
                return "\n".join(text)[::-1]
        return text

    def _apply_size_rule(self, size, attrs):
        if "size" in attrs:
            if attrs["size"] == "big":
                return size * 2
            if attrs["size"] == "small":
                return size * 0.25
        return size

    def _apply_highlight_rule(self, attrs, ax, text, x, y, size):
        if "highlight" in attrs:
            if attrs["highlight"] == "after":
                pos = x + (0.0035 * size * (len(text) / 2))
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.0005 * size)
                ab = AnnotationBbox(imagebox, (pos, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.0005 * size)
                ab = AnnotationBbox(imagebox, (pos, y - 0.12), frameon=False)
                ax.add_artist(ab)
            if attrs["highlight"] == "before":
                pos = x - (0.004 * size * (len(text) / 2))
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.0005 * size)
                ab = AnnotationBbox(imagebox, (pos, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.0005 * size)
                ab = AnnotationBbox(imagebox, (pos, y - 0.12), frameon=False)
                ax.add_artist(ab)
            if attrs["highlight"] == "middle":
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.0005 * size)
                ab = AnnotationBbox(imagebox, (x, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.0005 * size)
                ab = AnnotationBbox(imagebox, (x, y - 0.12), frameon=False)
                ax.add_artist(ab)

    def _apply_inside_rule(self, attrs, element):
        pass

    def render_inside_rule_puzzle(self, graph, show=False, save_path=""):
        edge_attrs = list(get_edge_information(graph).values())[0]
        text_inside = edge_attrs[0]["text"].upper()
        text_outside = edge_attrs[2]["text"].upper()
        if len(text_outside) % 2 == 1:
            text_outside_left = text_outside
            text_outside_right = text_outside
        else:
            text_outside_left = text_outside[:int(len(text_outside)/2)]
            text_outside_right = text_outside[int(len(text_outside)/2):]

        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))
        template = graph.graph["template"]["obj"]
        (x, y), size = template.elements[0][:2], template.elements[0][2] * 40
        text = ax.text(x, y, text_inside, color="black", ha="center", va="center", weight="bold",
                       fontsize=size, fontfamily="Consolas")
        ax.annotate(text_outside_left, xycoords=text, xy=(0, 0), va="bottom", ha="right", color="black", weight="bold", fontsize=size,
                    fontfamily="Consolas")
        ax.annotate(text_outside_right, xycoords=text, xy=(1, 0), va="bottom", color="black", weight="bold", fontsize=size,
                    fontfamily="Consolas")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        if save_path != "":
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)
