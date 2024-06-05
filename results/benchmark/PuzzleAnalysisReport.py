import json
import os
import glob

import networkx as nx
import pandas as pd

from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.patterns.Rule import Rule
from util import get_node_attributes


class PuzzleAnalysisReport:
    def __init__(self):
        self._compound_parser = CompoundRebusGraphParser()
        self._phrase_parser = PhraseRebusGraphParser()

    def generate_final(self):
        phrases = [os.path.basename(file).split(".")[0]
                   for file in glob.glob(f"{os.path.dirname(__file__)}/final/*")]
        ladec = pd.read_csv(f"{os.path.dirname(__file__)}/../../saved/ladec_raw_small.csv")
        custom_compounds = pd.read_csv(f"{os.path.dirname(__file__)}/../../saved/custom_compounds.csv")
        phrase_to_graph = {}
        for phrase in phrases:
            phrase_parts = phrase.split("_")
            phrase_index = 0
            if (phrase_parts[0] in ladec["stim"].tolist() and len(phrase_parts) == 2) or len(phrase_parts) == 1:
                if phrase_parts[-1].isnumeric():
                    phrase_index = int(phrase_parts[-1]) - 1
                    phrase_parts = phrase_parts[:-1]
                phrase_ = " ".join(phrase_parts)
                row = ladec.loc[ladec["stim"] == phrase_].values.flatten().tolist()
                if len(row) == 0:
                    row = custom_compounds.loc[custom_compounds["stim"] == phrase_].values.flatten().tolist()
                c1, c2, is_plural = row[0], row[1], bool(row[2])
                graphs = self._compound_parser.parse(c1, c2, is_plural)
                phrase_to_graph[phrase] = graphs[phrase_index]
            else:
                if phrase_parts[-1].isnumeric():
                    phrase_index = int(phrase_parts[-1]) - 1
                    phrase_parts = phrase_parts[:-1]
                phrase_ = " ".join(phrase_parts)
                graphs = self._phrase_parser.parse(phrase_)
                phrase_to_graph[phrase] = graphs[phrase_index]
        graphs = list(phrase_to_graph.values())
        self._count_rules(graphs)

    def generate_phrases(self):
        phrases = [os.path.basename(file).split(".")[0]
                   for file in glob.glob(f"{os.path.dirname(__file__)}/phrases/*")]
        phrase_to_graph = {}
        for phrase in phrases:
            phrase_parts = phrase.split("_")
            phrase_index = 0
            if phrase_parts[-1].isnumeric():
                phrase_index = int(phrase_parts[-1])-1
                phrase_parts = phrase_parts[:-1]
            phrase_ = " ".join(phrase_parts)
            graphs = self._phrase_parser.parse(phrase_)
            phrase_to_graph[phrase] = graphs[phrase_index]
        graphs = list(phrase_to_graph.values())
        self._count_rules(graphs)

    def _count_rules(self, graphs):
        rules = list(Rule.get_all_rules()["individual"].keys()) + ["sound"] + ["icon"]
        rules_freq_text, rules_freq_icon = {}, {}
        edge_freq, edge_freq_icon = {}, {}
        n_text_puzzles, n_icon_puzzles = 0, 0

        def increment_rule_freq(rule, value, contains_icons):
            if contains_icons:
                if rule in rules:
                    if rule not in rules_freq_icon:
                        rules_freq_icon[rule] = 0
                    rules_freq_icon[rule] += 1
                if f"{rule}_{value}" in rules:
                    if f"{rule}_{value}" not in rules_freq_icon:
                        rules_freq_icon[f"{rule}_{value}"] = 0
                    rules_freq_icon[f"{rule}_{value}"] += 1
            else:
                if rule in rules:
                    if rule not in rules_freq_text:
                        rules_freq_text[rule] = 0
                    rules_freq_text[rule] += 1
                if f"{rule}_{value}" in rules:
                    if f"{rule}_{value}" not in rules_freq_text:
                        rules_freq_text[f"{rule}_{value}"] = 0
                    rules_freq_text[f"{rule}_{value}"] += 1

        def format_attrs(attrs):
            attrs_ = attrs.copy()
            del attrs_["text"]
            if attrs_["repeat"] == 1:
                del attrs_["repeat"]
            for rule, value in attrs_.copy().items():
                if value == 2:
                    attrs_[rule] = "two"
                if value == 4:
                    attrs_[rule] = "four"
                if rule == "repeat":
                    attrs_["repetition"] = attrs_[rule]
                    del attrs_[rule]
            return attrs_

        for graph in graphs:
            node_attrs = get_node_attributes(graph)
            contains_icons = sum([1 if "icon" in attr else 0 for attr in node_attrs.values()]) > 0
            if contains_icons:
                n_icon_puzzles += 1
            else:
                n_text_puzzles += 1
            for node, attrs in node_attrs.items():
                attrs_ = format_attrs(attrs)
                for rule, value in attrs_.items():
                    increment_rule_freq(rule, value, contains_icons)
            edges = nx.get_edge_attributes(graph, "rule").values()
            for edge in edges:
                if not contains_icons:
                    if edge not in edge_freq:
                        edge_freq[edge] = 0
                    edge_freq[edge] += 1
                else:
                    if edge not in edge_freq_icon:
                        edge_freq_icon[edge] = 0
                    edge_freq_icon[edge] += 1

        rules_freq_text = dict(sorted(rules_freq_text.items()))
        rules_freq_icon = dict(sorted(rules_freq_icon.items()))
        del rules_freq_icon["icon"]

        print("=== BREAKDOWN: PUZZLES (TEXT) ===")
        print(f"Total puzzle frequency: {n_text_puzzles}")
        print(f"Total rule frequency: {sum(rules_freq_text.values())}")
        print(json.dumps(rules_freq_text, indent=3))
        print(json.dumps(edge_freq, indent=3))

        print(f"=== BREAKDOWN PUZZLES (ICONS) ===")
        print(f"Total puzzle frequency: {n_icon_puzzles}")
        print(f"Total rule frequency: {sum(rules_freq_icon.values())}")
        print(json.dumps(rules_freq_icon, indent=3))
        print(json.dumps(edge_freq_icon, indent=3))
