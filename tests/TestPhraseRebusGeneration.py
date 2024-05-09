import unittest

from graphs.RebusGraph import RebusGraph
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.RebusImageConverterV2 import RebusImageConverterV2
from graphs.templates.Template import Template


class TestPhraseRebusGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()
        self.generator = RebusImageConverterV2()

    def test_clean_up_ones_act(self):
        phrase = "clean up one's act"
        graph = self.parser.parse(phrase)[0]
        self.generator.generate(graph, show=True)

    def test_right_back_at_you(self):
        phrase = "right back at you"
        graph = self.parser.parse(phrase)[1]
        print(graph)
        self.generator.generate(graph, show=True)

    def test_back_to_the_wall(self):
        phrase = "back to the wall"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            self.generator.generate(graph, show=True)

