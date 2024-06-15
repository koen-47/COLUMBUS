import json
import os
import glob

import networkx as nx
import numpy as np
import pandas as pd

from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.patterns.Rule import Rule
from util import get_node_attributes, get_answer_graph_pairs


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

    def compute_basic_statistics(self):
        phrase_graphs, compound_graphs = get_answer_graph_pairs()
        graphs = {}
        graphs.update(compound_graphs)
        graphs.update(phrase_graphs)

        def calculate_number_of_graphs_n_nodes(graphs, n):
            return np.array([1 for graph in graphs if graph.number_of_nodes() == n]).sum()

        graphs_no_icons, graphs_icons = {}, {}
        for answer, graph in graphs.items():
            contains_icons = sum([1 if "icon" in attr else 0 for attr in get_node_attributes(graph).values()]) > 0
            if contains_icons:
                graphs_icons[answer] = graph
            else:
                graphs_no_icons[answer] = graph

        answers = [" ".join(answer.split("_")[:-1]) if answer.split("_")[-1].isnumeric() else " ".join(answer.split("_"))
                   for answer in graphs.keys()]
        avg_answer_len = np.array([len(answer.split()) for answer in answers]).mean()

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

        print("\n=== BENCHMARK STATISTICS (overall) ===")
        print("Number of puzzles:", len(graphs))
        print("Number of single node graphs:", n_single_node_graphs)
        print("Number of double node graphs:", n_double_node_graphs)
        print("Number of triple node graphs:", n_triple_node_graphs)
        print("Avg. number of nodes per graph:", avg_n_nodes)
        print("Avg. number of edges per graph", avg_n_edges)

        print("\n=== BENCHMARK STATISTICS (no icons) ===")
        print("Number of puzzles (no icons):", len(graphs_no_icons))
        print("Number of single node graphs (no icons):", n_single_node_graphs_no_icons)
        print("Number of double node graphs (no icons):", n_double_node_graphs_no_icons)
        print("Number of triple node graphs (no icons):", n_triple_node_graphs_no_icons)
        print("Avg. number of nodes per graph (no icons):", avg_n_nodes_no_icon)
        print("Avg. number of edges per graph (no icons)", avg_n_edges_no_icon)

        print("\n=== BENCHMARK STATISTICS (icons) ===")
        print("Number of puzzles (icons):", len(graphs_icons))
        print("Number of single node graphs (icons):", n_single_node_graphs_icons)
        print("Number of double node graphs (icons):", n_double_node_graphs_icons)
        print("Number of triple node graphs (icons):", n_triple_node_graphs_icons)
        print("Avg. number of nodes per graph (icons):", avg_n_nodes_icon)
        print("Avg. number of edges per graph (icons)", avg_n_edges_icon)

        with open("./saved/distractors_all-minilm-l6-v2_final.json", "r") as file:
            distractors = json.load(file)
            visible_words = {answer: " ".join([node["text"].lower() for node in get_node_attributes(graph).values()]) for answer, graph in graphs.items()}
            print(len(distractors), distractors)
            print(len(visible_words), visible_words)

            compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in pd.read_csv("./saved/ladec_raw_small.csv").iterrows()}
            custom_compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in pd.read_csv("./saved/custom_compounds.csv").iterrows()}
            compounds.update(custom_compounds)

            print(len(visible_words), visible_words)

            distractors = {compounds[answer]: distractor for answer, distractor in distractors.items() if answer in compounds}
            # distractors = {(" ".join(answer.split("_")[:-1]) if answer.split("_")[-1].isnumeric() else " ".join(answer.split("_"))): distractor
            #                for answer, distractor in distractors.items()}



        # print("Number of compounds:", len(compound_graphs))
        # print("Number of phrases:", len(phrase_graphs))
        # print("Avg. answer length:", avg_answer_len)
        # print(f"Number of nodes in a graph: [1: {n_single_node_graphs}, 2: {n_two_node_graphs}, "
        #       f"3: {n_three_node_graphs}, 4: {n_four_node_graphs}]")

