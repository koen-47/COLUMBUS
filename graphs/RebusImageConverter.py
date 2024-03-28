from itertools import cycle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import ConnectionPatch

from util import get_node_attributes, get_edges_from_node, get_edge_information
from .templates.Template import Template


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
        text = attrs["text"]
        text = self._apply_direction_rule(text, attrs)
        color = self._apply_color_rule(attrs)
        ha = self._apply_position_ha_rule(attrs)
        size = self._apply_size_rule(size, attrs)
        ax.text(x, y, text, fontsize=size, fontweight="bold", fontfamily="Consolas", color=color, ha=ha,
                va="center")
        self._apply_cross_rule(attrs, ax, text, x, y, size)
        self._apply_highlight_rule(attrs, ax, text, x, y, size)

    def _apply_color_rule(self, attrs):
        return "black" if "color" not in attrs else attrs["color"]

    def _apply_cross_rule(self, attrs, ax, text, x, y, size):
        if "cross" in attrs:
            line_x1, line_x2 = x - (0.0025 * size * (len(text) / 2)), x + (0.0025 * size * (len(text) / 2))
            line = ConnectionPatch((line_x1, y), (line_x2, y), "axes fraction", "axes fraction",
                                   color="black", lw=2)
            ax.add_artist(line)

    def _apply_position_ha_rule(self, attrs):
        if "position" in attrs:
            if attrs["position"] == "left":
                return "left"
            if attrs["position"] == "right":
                return "right"
        return "center"

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
                imagebox = OffsetImage(arrow_top, zoom=0.025)
                ab = AnnotationBbox(imagebox, (pos, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.025)
                ab = AnnotationBbox(imagebox, (pos, y - 0.12), frameon=False)
                ax.add_artist(ab)
            if attrs["highlight"] == "before":
                pos = x - (0.004 * size * (len(text) / 2))
                arrow_top = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 3)
                imagebox = OffsetImage(arrow_top, zoom=0.025)
                ab = AnnotationBbox(imagebox, (pos, y + 0.15), frameon=False)
                ax.add_artist(ab)
                arrow_bottom = np.rot90(plt.imread("./saved/resources/arrow_right.png"), 1)
                imagebox = OffsetImage(arrow_bottom, zoom=0.025)
                ab = AnnotationBbox(imagebox, (pos, y - 0.12), frameon=False)
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

    def render_inside_rule_puzzle(self, graph):
        edge_attrs = list(get_edge_information(graph).values())[0]
        text_inside = edge_attrs[0]["text"].upper()
        text_outside = edge_attrs[2]["text"].upper()
        text_outside_left, text_outside_right = "", ""
        if len(text_outside) % 2 == 1:
            text_outside_left = text_outside
            text_outside_right = text_outside
        else:
            text_outside_left = text_outside[:int(len(text_outside)/2)]
            text_outside_right = text_outside[int(len(text_outside)/2):]

        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))
        template = graph.graph["template"]["obj"]
        (x, y), size = template.elements[0]["singular"][:2], template.elements[0]["singular"][2] * 40
        text = ax.text(x, y, text_inside, color="black", ha="center", va="center", weight="bold",
                       fontsize=size, fontfamily="Consolas")
        ax.annotate(text_outside_left, xycoords=text, xy=(0, 0), va="bottom", ha="right", color="black", weight="bold", fontsize=size,
                    fontfamily="Consolas")
        ax.annotate(text_outside_right, xycoords=text, xy=(1, 0), va="bottom", color="black", weight="bold", fontsize=size,
                    fontfamily="Consolas")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.show()
        # if save_path != "":
        #     plt.savefig(save_path)
        # if show:
        #     plt.show()
        # plt.close(fig)
