import json
import os
import glob

import networkx as nx
import numpy as np
import pandas as pd

from puzzles.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from puzzles.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from puzzles.patterns.Rule import Rule
from util import get_node_attributes, get_answer_graph_pairs


class PuzzleAnalysisReport:
    def __init__(self):
        self._compound_parser = CompoundRebusGraphParser()
        self._phrase_parser = PhraseRebusGraphParser()

    def generate(self):
        self._compute_basic_statistics()
        graphs = list(get_answer_graph_pairs("v3", combine=True).values())
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
        rules_freq_icon = {rule: ("-" if str(rule).startswith("direction") else freq) for rule, freq in
                           rules_freq_icon.items()}
        del rules_freq_icon["icon"]

        print("\n === BREAKDOWN: PUZZLE RULES ===")
        print(f"Total puzzle frequency (no icons): {n_text_puzzles}")
        print(f"Total rule frequency (no icons): {sum(rules_freq_text.values())}")
        print(f"Total puzzle frequency (icons): {n_icon_puzzles}")
        print(f"Total rule frequency (icons): {sum([freq for freq in rules_freq_icon.values() if freq != '-'])}")

        print("\nIndividual + Modifier Rule Frequency Table")
        print(pd.DataFrame({"no_icons": rules_freq_text, "icons": rules_freq_icon}))

        print("\nRelational Rule Frequency Table")
        print(pd.DataFrame({"no_icons": edge_freq, "icons": edge_freq_icon}))

    def _compute_basic_statistics(self):
        phrase_graphs, compound_graphs = get_answer_graph_pairs(version="v3")
        graphs = {}
        graphs.update(compound_graphs)
        graphs.update(phrase_graphs)

        def calculate_number_of_graphs_n_nodes(graphs, n):
            return np.array([1 for graph in graphs if graph.number_of_nodes() == n]).sum()

        graphs_no_icons, graphs_icons = {}, {}
        answers_no_icons, answers_icons = [], []
        for answer, graph in graphs.items():
            answer_ = " ".join(answer.split("_")[:-1]) if answer.split("_")[-1].isnumeric() else " ".join(
                answer.split("_"))
            contains_icons = sum([1 if "icon" in attr else 0 for attr in get_node_attributes(graph).values()]) > 0
            if contains_icons:
                graphs_icons[answer] = graph
                answers_icons.append(answer_)
            else:
                graphs_no_icons[answer] = graph
                answers_no_icons.append(answer_)
        answers = [
            " ".join(answer.split("_")[:-1]) if answer.split("_")[-1].isnumeric() else " ".join(answer.split("_"))
            for answer in graphs.keys()]

        # for answer, graph in puzzles.items():
        #     if graph.number_of_nodes() == 4:
        #         print(answer)

        avg_answer_len = np.array([len(answer.split()) for answer in answers]).mean()
        avg_answer_len_no_icon = np.array([len(answer.split()) for answer in answers_no_icons]).mean()
        avg_answer_len_icon = np.array([len(answer.split()) for answer in answers_icons]).mean()

        avg_n_nodes = np.array([graph.number_of_nodes() for graph in graphs.values()]).mean()
        avg_n_edges = np.array([graph.number_of_edges() for graph in graphs.values()]).mean()
        n_single_node_graphs = calculate_number_of_graphs_n_nodes(graphs.values(), n=1)
        n_double_node_graphs = calculate_number_of_graphs_n_nodes(graphs.values(), n=2)
        n_triple_node_graphs = calculate_number_of_graphs_n_nodes(graphs.values(), n=3)

        avg_n_nodes_no_icon = np.array([graph.number_of_nodes() for graph in graphs_no_icons.values()]).mean()
        avg_n_edges_no_icon = np.array([graph.number_of_edges() for graph in graphs_no_icons.values()]).mean()
        n_single_node_graphs_no_icons = calculate_number_of_graphs_n_nodes(graphs_no_icons.values(), n=1)
        n_double_node_graphs_no_icons = calculate_number_of_graphs_n_nodes(graphs_no_icons.values(), n=2)
        n_triple_node_graphs_no_icons = calculate_number_of_graphs_n_nodes(graphs_no_icons.values(), n=3)

        avg_n_nodes_icon = np.array([graph.number_of_nodes() for graph in graphs_icons.values()]).mean()
        avg_n_edges_icon = np.array([graph.number_of_edges() for graph in graphs_icons.values()]).mean()
        n_single_node_graphs_icons = calculate_number_of_graphs_n_nodes(graphs_icons.values(), n=1)
        n_double_node_graphs_icons = calculate_number_of_graphs_n_nodes(graphs_icons.values(), n=2)
        n_triple_node_graphs_icons = calculate_number_of_graphs_n_nodes(graphs_icons.values(), n=3)

        with open("./saved/distractors_v3.json", "r") as file:
            answer_to_distractors = json.load(file)

            visible_to_distractor = {}
            for answer, distractors in answer_to_distractors.items():
                visible_words = []
                node_attrs = get_node_attributes(graphs[answer])
                contains_icons = sum([1 if "icon" in attr else 0 for attr in node_attrs.values()]) > 0
                for attr in node_attrs.values():
                    word = attr["text"].lower()
                    if "icon" in attr:
                        word = list(attr["icon"].keys())[0]
                    visible_words.append(word)
                visible_words = " ".join(visible_words)
                visible_to_distractor[visible_words] = {"distractors": answer_to_distractors[answer],
                                                        "contains_icon": contains_icons}

            compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in
                         pd.read_csv("./saved/ladec_raw_small.csv").iterrows()}
            custom_compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in
                                pd.read_csv("./saved/custom_compounds.csv").iterrows()}
            compounds.update(custom_compounds)

            for visible, distractors in visible_to_distractor.items():
                visible_to_distractor[visible]["distractors"] = [
                    compounds[distractor] if distractor in compounds else distractor
                    for distractor in distractors["distractors"]]

            for visible, distractors in visible_to_distractor.items():
                word_overlaps = []
                for distractor in distractors["distractors"]:
                    visible_set, distractor_set = set(visible.split()), set(distractor.split())
                    overlap = len(visible_set.intersection(distractor_set))
                    word_overlaps.append(overlap)
                word_overlaps = np.array(word_overlaps).sum()
                visible_to_distractor[visible]["word_overlap"] = word_overlaps

            avg_visible_word_overlap = np.array([distractors["word_overlap"] for distractors in
                                                      visible_to_distractor.values()]).mean()

            pct_contains_visible = np.array([1 if distractors["word_overlap"] > 0 else 0 for distractors
                                                  in visible_to_distractor.values()]).sum() / len(visible_to_distractor)

            avg_visible_word_overlap_no_icon = np.array([distractors["word_overlap"] for distractors in
                                                         visible_to_distractor.values() if not
                                                         distractors["contains_icon"]]).mean()

            pct_contains_visible_no_icon = np.array([1 if distractors["word_overlap"] > 0 else 0 for
                                                     distractors in visible_to_distractor.values() if
                                                     not distractors["contains_icon"]]).sum() / len(graphs_no_icons)

            avg_visible_word_overlap_icon = np.array([distractors["word_overlap"] for distractors in
                                                      visible_to_distractor.values() if
                                                      distractors["contains_icon"]]).mean()

            pct_contains_visible_icon = np.array([1 if distractors["word_overlap"] > 0 else 0 for distractors
                                                  in visible_to_distractor.values() if
                                                  distractors["contains_icon"]]).sum() / len(graphs_icons)

            print(f"Avg. visible word overlap: {avg_visible_word_overlap_no_icon}")
            print(f"% of distractors that contain at least one visible word: {pct_contains_visible_no_icon}")

        print("\n=== BENCHMARK STATISTICS (overall) ===")
        print("Number of puzzles:", len(graphs))
        print(f"Avg. answer length", avg_answer_len)
        print("Number of single node puzzles:", n_single_node_graphs)
        print("Number of double node puzzles:", n_double_node_graphs)
        print("Number of triple node puzzles:", n_triple_node_graphs)
        print("Avg. number of nodes per graph:", avg_n_nodes)
        print("Avg. number of edges per graph", avg_n_edges)
        print(f"Avg. visible word overlap: {avg_visible_word_overlap}")
        print(f"% of distractors that contain at least one visible word (no icons): {pct_contains_visible}")

        print("\n=== BENCHMARK STATISTICS (no icons) ===")
        print("Number of puzzles (no icons):", len(graphs_no_icons))
        print(f"Avg. answer length (no icons)", avg_answer_len_no_icon)
        print("Number of single node puzzles (no icons):", n_single_node_graphs_no_icons)
        print("Number of double node puzzles (no icons):", n_double_node_graphs_no_icons)
        print("Number of triple node puzzles (no icons):", n_triple_node_graphs_no_icons)
        print("Avg. number of nodes per graph (no icons):", avg_n_nodes_no_icon)
        print("Avg. number of edges per graph (no icons)", avg_n_edges_no_icon)
        print(f"Avg. visible word overlap (no icons): {avg_visible_word_overlap_no_icon}")
        print(f"% of distractors that contain at least one visible word (no icons): {pct_contains_visible_no_icon}")

        print("\n=== BENCHMARK STATISTICS (icons) ===")
        print("Number of puzzles (icons):", len(graphs_icons))
        print(f"Avg. answer length (icon)", avg_answer_len_icon)
        print("Number of single node puzzles (icons):", n_single_node_graphs_icons)
        print("Number of double node puzzles (icons):", n_double_node_graphs_icons)
        print("Number of triple node puzzles (icons):", n_triple_node_graphs_icons)
        print("Avg. number of nodes per graph (icons):", avg_n_nodes_icon)
        print("Avg. number of edges per graph (icons)", avg_n_edges_icon)
        print(f"Avg. visible word overlap (icons): {avg_visible_word_overlap_icon}")
        print(f"% of distractors that contain at least one visible word (icons): {pct_contains_visible_icon}")
