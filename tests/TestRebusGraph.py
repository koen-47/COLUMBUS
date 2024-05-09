import json
import os
import unittest

from tqdm import tqdm

from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser


class TestRebusGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhraseRebusGraphParser()

    def test_is_invalid_to_generate_1(self):
        graph = self.parser.parse("clean up one's act")[0]
        self.assertIsNotNone(graph)

    def test_is_invalid_to_generate_2(self):
        graph = self.parser.parse("in the mood")
        self.assertIsNone(graph)

    def test_count_rules_1(self):
        graph = self.parser.parse("clean up one's act")[0]
        n_ind_rules, n_rel_rules = graph.compute_difficulty()
        self.assertEqual(n_ind_rules, 2)
        self.assertEqual(n_rel_rules, 0)

    def test_count_graph_difficulty(self):
        with open(f"{os.path.dirname(__file__)}/../saved/idioms_raw.json", "r") as file:
            idioms = json.load(file)
            difficulty_freq = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            idiom_to_difficulty = {}
            for idiom in tqdm(idioms, desc="Computing difficulty (phrases)"):
                graphs = self.parser.parse(idiom)
                if graphs is None:
                    continue
                for graph in graphs:
                    n_rules = sum(list(graph.compute_difficulty()))
                    if n_rules in difficulty_freq:
                        difficulty_freq[n_rules] += 1
                    if n_rules != 0:
                        if idiom not in idiom_to_difficulty:
                            idiom_to_difficulty[idiom] = []
                        idiom_to_difficulty[idiom].append(n_rules)
            idiom_to_difficulty = {idiom: max(difficulty) for idiom, difficulty in idiom_to_difficulty.items()}
            idiom_to_difficulty = dict(sorted(idiom_to_difficulty.items(), key=lambda item: item[1], reverse=True))

            print("# >=1.5:", len([difficulty for difficulty in idiom_to_difficulty.values() if difficulty >= 1.5]))
            print("# >=1.25:", len([difficulty for difficulty in idiom_to_difficulty.values() if difficulty >= 1.25]))
            print("# >=1.0:", len([difficulty for difficulty in idiom_to_difficulty.values() if difficulty >= 1.]))
            print(json.dumps(idiom_to_difficulty, indent=3))
