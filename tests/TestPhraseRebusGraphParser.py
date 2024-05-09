import unittest

from graphs.RebusGraph import RebusGraph
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.templates.Template import Template


class TestPhraseRebusGraphParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()

    def test_pull_out_of_the_fire(self):
        graphs = self.parser.parse("pull out of the fire")
        for graph in graphs:
            print(graph)

    def test_rub_off_on(self):
        graphs = self.parser.parse("rub off on")
        for graph in graphs:
            print(graph)

    def test_take_a_bite_out_of(self):
        graphs = self.parser.parse("beat the stuffing out of")
        print(graphs)
