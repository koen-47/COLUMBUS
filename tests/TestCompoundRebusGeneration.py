import unittest

from graphs.RebusImageConverterV2 import RebusImageConverterV2
from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser


class TestCompoundRebusGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = CompoundRebusGraphParser()
        self.generator = RebusImageConverterV2()

    def test_color_blue(self):
        self.generate("blue", "berry", is_plural=False)
        self.generate("blue", "berry", is_plural=True)

    def test_highlight(self):
        self.generate("before", "dawn", is_plural=False)
        self.generate("before", "dawns", is_plural=True)
        self.generate("mid", "night", is_plural=False)
        self.generate("mid", "nights", is_plural=True)
        self.generate("after", "thought", is_plural=False)
        self.generate("after", "thoughts", is_plural=True)

    def test_direction(self):
        self.generate("up", "draft", is_plural=False)
        self.generate("up", "drafts", is_plural=True)
        self.generate("break", "down", is_plural=False)
        self.generate("break", "downs", is_plural=True)
        self.generate("back", "down", is_plural=False)
        self.generate("back", "downs", is_plural=True)

    def test_cross(self):
        self.generate("cross", "road", is_plural=False)
        self.generate("cross", "roads", is_plural=True)

    def test_size(self):
        self.generate("grand", "father", is_plural=False)   
        self.generate("grand", "fathers", is_plural=True)
        self.generate("micro", "chip", is_plural=False)
        self.generate("micro", "chips", is_plural=True)

    def test_icon(self):
        self.generate("red", "coat", is_plural=False)

    def generate(self, c1, c2, is_plural):
        graphs = self.parser.parse(c1, c2, is_plural=is_plural)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)
