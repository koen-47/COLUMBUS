import unittest

from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser


class TestPhraseRebusGraphParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()

    def test_pull_out_of_the_fire(self):
        graphs = self.parser.parse("pull out of the fire")
        for graph in graphs:
            print(graph)

    def test_aftereffects(self):
        graphs = self.parser.parse("aftereffects")
        for graph in graphs:
            print(graph)

    def test_pull_wool_over_eyes(self):
        graphs = self.parser.parse("pull wool over eyes")
        for graph in graphs:
            print(graph)

    def test_cross_fingers_for_luck(self):
        graphs = self.parser.parse("cross fingers for luck")
        for graph in graphs:
            print(graph)

    def test_rub_off_on(self):
        graphs = self.parser.parse("rub off on")
        for graph in graphs:
            print(graph)

    def test_take_a_bite_out_of(self):
        graphs = self.parser.parse("beat the stuffing out of")
        print(graphs)

    def test_whats_in_it_for_me(self):
        graphs = self.parser.parse("what's in it for me")
        print(graphs)

    def test_cross_my_heart_and_hope_to_die(self):
        graphs = self.parser.parse("cross my heart and hope to die")
        print(graphs)

    def test_go_big_or_go_home(self):
        graphs = self.parser.parse("go big or go home")
        for graph in graphs:
            print(graph)

    def test_off_ones_cross(self):
        graphs = self.parser.parse("cross one's heart")
        for graph in graphs:
            print(graph)

    def test_stars_in_ones_eyes(self):
        graphs = self.parser.parse("stars in one's eyes")
        for graph in graphs:
            print(graph)

    def test_eyes_on_the_prize(self):
        graphs = self.parser.parse("eyes on the prize")
        for graph in graphs:
            print(graph)

    def test_put_on_a_red_light(self):
        graphs = self.parser.parse("put on a red light")
        for graph in graphs:
            print(graph)

    def test_to_the_stars(self):
        graphs = self.parser.parse("to the stars")
        for graph in graphs:
            print(graph)

    def test_blow_up_in_ones_face(self):
        graphs = self.parser.parse("blow up in one's face")
        for graph in graphs:
            print(graph)

