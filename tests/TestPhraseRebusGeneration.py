import json
import os
import unittest

from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.RebusImageConverterV2 import RebusImageConverterV2
from graphs.patterns.Rule import Rule


class TestPhraseRebusGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()
        self.generator = RebusImageConverterV2()

    def test_clean_up_ones_act(self):
        phrase = "clean up one's act"
        graph = self.parser.parse(phrase)[0]
        self.generator.generate(graph, show=True)

    def test_loss_for_words(self):
        phrase = "at a loss for words"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            self.generator.generate(graph, show=True)

    def test_bring_down_the_house(self):
        phrase = "bring down the house"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            self.generator.generate(graph, show=True)

    def test_cross_icons(self):
        phrase = "cross the T's and dot the I's"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            self.generator.generate(graph, show=True)
        phrase = "cross over to the other side"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            self.generator.generate(graph, show=True)
        phrase = "cross fingers for luck"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            self.generator.generate(graph, show=True)

    def test_birds_eye_view(self):
        phrase = "bird's eye view"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)

    def test_cross_bow(self):
        phrase = "cross bow"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)

    def test_running_far_behind(self):
        phrase = "far behind"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)

    def test_come_down_to_earth(self):
        phrase = "come down to earth"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)

    def test_go_big_or_go_home(self):
        phrase = "go big or go home"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            # print(graph)
            self.generator.generate(graph, show=True)
        phrase = "to cast a large shadow"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            # print(graph)
            self.generator.generate(graph, show=True)

    def test_one_act(self):
        phrase = "one"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
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

    def test_cross_my_heart_and_hope_to_die(self):
        phrase = "cross my heart and hope to die"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)

    def test_icon(self):
        graphs = self.parser.parse("horse for you")
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)
        graphs = self.parser.parse("eyes for you")
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)
        graphs = self.parser.parse("stand for")
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)


class TestPhraseGenerationInside(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()
        self.generator = RebusImageConverterV2()

    def test_inside_icon(self):
        phrase = "stars in one's eyes"
        self.generate_phrase(phrase)
        phrase = "devil is in the detail"
        self.generate_phrase(phrase)

    def test_inside_size(self):
        phrase = "big fish in a small pond"
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate(graph, show=True)

    def test_inside_repeat_two(self):
        phrase = "to hell in a handbasket"
        self.generate_phrase(phrase)

    def test_inside_direction_up(self):
        phrase = "blow up in one's face"
        self.generate_phrase(phrase)

    def test_inside_color_outside(self):
        phrase = "once in a blue moon"
        self.generate_phrase(phrase)

    def test_inside_two_words(self):
        phrase = "beauty is in the eye of the beholder"
        self.generate_phrase(phrase)

    def test_inside_back(self):
        phrase = "go back in time"
        self.generate_phrase(phrase)

    def test_inside_up(self):
        phrase = "go up in smoke"
        self.generate_phrase(phrase)
        phrase = "blow up in one's face"
        self.generate_phrase(phrase)

    def test_inside_down(self):
        phrase = "go down in flames"
        self.generate_phrase(phrase)

    def test_inside_repeat_four(self):
        phrase = "one in the eye for"
        self.generate_phrase(phrase)
        # phrase = "what's in it for me"
        # self.generate_phrase(phrase)

    def generate_phrase(self, phrase):
        graphs = self.parser.parse(phrase)
        for graph in graphs:
            print(graph)
            self.generator.generate_inside(graph, show=True)
        # best_graph, best_quality = max([(graph, graph.compute_difficulty()) for graph in graphs], key=lambda x: x[1])
        # print(best_graph, best_quality)
        # self.generator.generate_inside(best_graph, show=True)

    def test_generate_all(self):
        counter = 0
        with open(f"{os.path.dirname(__file__)}/../saved/idioms_raw.json", "r") as file:
            idioms = json.load(file)
            inside_keywords = Rule.Relational.INSIDE
            for idiom in idioms:
                if len(set(inside_keywords).intersection(set(idiom.split()))) >= 1:
                    graphs = self.parser.parse(idiom)
                    if graphs is not None:
                        counter += 1
                        print(f"{counter}) {idiom}")
                        for graph in graphs:
                            self.generator.generate_inside(graph, show=True)
        print(counter)


class TestPhraseGenerationAbove(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()
        self.generator = RebusImageConverterV2()

    def test_above_basic(self):
        phrase = "head above water"
        self.generate_phrase(phrase)
        phrase = "easy on the eyes"
        self.generate_phrase(phrase)

    def test_icon_below(self):
        phrase = "eyes on the prize"
        self.generate_phrase(phrase)

    def test_above_sound(self):
        phrase = "once upon a time"
        self.generate_phrase(phrase)

    def test_below_long(self):
        phrase = "punch above one's weight"
        self.generate_phrase(phrase)
        phrase = "rest on one's laurels"
        self.generate_phrase(phrase)

    def test_below_repeat(self):
        phrase = "go over to the majority"
        self.generate_phrase(phrase)
        phrase = "try on for size"
        self.generate_phrase(phrase)

    def test_below_icon(self):
        phrase = "eyes on the prize"
        self.generate_phrase(phrase)

    def test_below_icon_2(self):
        phrase = "put cards on the table"
        self.generate_phrase(phrase)

    def test_below_color_icon(self):
        phrase = "put on a red light"
        self.generate_phrase(phrase)

    # def test_simple(self):
    #     phrase = "punch above one's weight"
    #     graphs = self.parser.parse(phrase)[0]
    #     self.generator.is_simple(graphs)

    def test_generate_all(self):
        with open(f"{os.path.dirname(__file__)}/../saved/idioms_raw.json", "r") as file:
            idioms = json.load(file)
            above_keywords = Rule.Relational.ABOVE
            for idiom in idioms:
                if len(set(above_keywords).intersection(set(idiom.split()))) >= 1:
                    graphs = self.parser.parse(idiom)
                    if graphs is not None:
                        print(idiom)
                        for graph in graphs:
                            try:
                                self.generator.generate_above(graph, show=True)
                            except:
                                continue

    def generate_phrase(self, phrase):
        graphs = self.parser.parse(phrase)
        # best_graph, best_quality = max([(graph, graph.compute_difficulty()) for graph in graphs], key=lambda x: x[1])
        # print(best_graph, best_quality)
        # self.generator.generate_above(best_graph, show=True)
        print(graphs)
        for graph in graphs:
            print(graph)
            self.generator.generate_above(graph, show=True)


class TestPhraseGenerationOutside(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()
        self.generator = RebusImageConverterV2()

    def test_outside_basic(self):
        phrase = "think outside the box"
        self.generate_phrase(phrase)

    def test_fall_out_with(self):
        phrase = "fall out with"
        self.generate_phrase(phrase)

    def test_outside_icon(self):
        phrase = "pull out of the fire"
        self.generate_phrase(phrase)

    def test_outside_repeat_four(self):
        phrase = "for crying out loud"
        self.generate_phrase(phrase)

    def generate_phrase(self, phrase):
        graphs = self.parser.parse(phrase)
        best_graph, best_quality = max([(graph, graph.compute_difficulty()) for graph in graphs], key=lambda x: x[1])
        print(best_graph, best_quality)
        self.generator.generate_outside(best_graph, show=True)

    def test_generate_all(self):
        counter = 0
        with open(f"{os.path.dirname(__file__)}/../saved/idioms_raw.json", "r") as file:
            idioms = json.load(file)
            outside_keywords = Rule.Relational.OUTSIDE
            for idiom in idioms:
                if len(set(outside_keywords).intersection(set(idiom.split()))) >= 1:
                    graphs = self.parser.parse(idiom)
                    if graphs is not None:
                        counter += 1
                        print(f"{counter}) {idiom}")
                        for graph in graphs:
                            self.generator.generate_outside(graph, show=True)
        print(counter)
