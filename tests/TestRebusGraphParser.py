import unittest

import networkx as nx

from graphs.RebusGraph import RebusGraph
from graphs.RebusGraphParser import RebusGraphParser


class TestRebusGraphParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = RebusGraphParser("../saved/ladec_raw_small.csv")

    def test_rebus_parse_1(self):
        graph = self.parser.parse_compound("redcoats")[0]
        correct_graph = RebusGraph()
        correct_graph.add_node(1, text="COAT", color="red")
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))

    def test_rebus_parser_2(self):
        graph_1, graph_2 = self.parser.parse_compound("greenback")
        correct_graph_1 = RebusGraph()
        correct_graph_1.add_node(1, text="BACK", color="green")
        correct_graph_2 = RebusGraph()
        correct_graph_2.add_node(1, text="GREEN", reverse=True)
        self.assertTrue(nx.utils.graphs_equal(graph_1, correct_graph_1))
        self.assertTrue(nx.utils.graphs_equal(graph_2, correct_graph_2))

    def test_rebus_parser_3(self):
        graph = RebusGraph()
        graph.add_node(1, text="BASE")
        graph = self.parser.parse_compound("blueberry", graph=graph)[0]
        correct_graph = RebusGraph()
        correct_graph.add_node(1, text="BASE")
        correct_graph.add_node(2, text="BERRY", color="blue")
        self.assertTrue(nx.utils.graphs_equal(graph, correct_graph))
