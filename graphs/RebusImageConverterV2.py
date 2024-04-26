import matplotlib.pyplot as plt

from util import get_node_attributes
from .templates.Template import Template


class RebusImageConverterV2:
    def __init__(self):
        self.BASE_SIZE = (400, 400)

    def generate(self, graph):
        fig, ax = plt.subplots(figsize=(self.BASE_SIZE[0] / 100, self.BASE_SIZE[1] / 100))

        node_attrs = get_node_attributes(graph)

        template = self._select_template(graph)

        max_chars = max([len(node["text"]) for node in list(node_attrs.values())])
        print(max_chars)

        for element, (node, attrs) in zip(template.elements, node_attrs.items()):
            print(element, attrs)
            # if attrs["text"] == "CLEAN":
            #     attrs["text"] = "\n".join(attrs["text"])
            ax.text(element[0], element[1], attrs["text"], fontsize=35, fontweight="bold", fontfamily="Consolas",
                    ha="center", va="center")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        # plt.show()

    def _select_template(self, graph):
        n_nodes = len(graph.nodes)
        if n_nodes == 3:
            return Template.BASE_THREE
