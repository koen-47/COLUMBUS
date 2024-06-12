import unittest

import networkx as nx

from graphs.RebusGraph import RebusGraph
from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.templates.Template import Template


class TestCompoundRebusGraphParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = CompoundRebusGraphParser()

    def test_rebus_parse_direction_down(self):
        compound = ["break", "downs", True]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="BREAK", direction="down", repeat=2)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_direction_up(self):
        compound = ["break", "up", True]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="BREAK", direction="up", repeat=2)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_direction_reverse(self):
        compound = ["back", "gammon", False]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="GAMMON", direction="reverse", repeat=1)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_style_color(self):
        compound = ["red", "coat", False]
        for graph in self.parser.parse(*compound):
            print(graph)
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="COAT", color="red", repeat=1)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_style_color_2(self):
        compound = ["red", "coats", True]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="COAT", color="red", repeat=2)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_style_cross(self):
        compound = ["cross", "roads", True]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="ROAD", cross=True, repeat=2)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_style_size_big(self):
        compound = ["big", "wigs", True]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="WIG", size="big", repeat=2)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_style_size_small(self):
        compound = ["small", "pox", False]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="POX", size="small", repeat=1)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_highlight_before(self):
        compound = ["before", "dawn", False]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="DAWN", highlight="before", repeat=1)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_highlight_middle(self):
        compound = ["mid", "night", False]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="NIGHT", highlight="middle", repeat=1)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_highlight_after(self):
        compound = ["after", "shocks", True]
        graph = self.parser.parse(*compound)[0]
        correct_graph = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph.add_node(1, text="SHOCK", highlight="after", repeat=2)
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parse_repetition(self):
        compound = ["to", "for", False]
        graphs = self.parser.parse(*compound)

        graph_1, graph_2 = graphs[0], graphs[1]
        correct_graph_1 = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph_1.add_node(1, text="4", repeat=2, sound={"to": ["2"]})
        correct_graph_2 = RebusGraph(template={"name": "base", "obj": Template.BASE})
        correct_graph_2.add_node(1, text="2", repeat=4, sound={"for": ["4"]})

        self.assertTrue(nx.utils.graphs_equal(graph_1, correct_graph_1))
        self.assertTrue(nx.utils.graphs_equal(graph_2, correct_graph_2))

