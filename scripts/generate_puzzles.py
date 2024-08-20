"""
Code to generate the puzzles for phrases and compounds.
"""

import json
import os

import networkx as nx
import pandas as pd
from tqdm import tqdm
from wordfreq import word_frequency

from puzzles.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from puzzles.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from puzzles.RebusImageConverter import RebusImageConverter
from util import get_node_attributes, get_answer_graph_pairs
from puzzles.Benchmark import Benchmark

phrase_parser = PhraseRebusGraphParser()
compound_parser = CompoundRebusGraphParser()
generator = RebusImageConverter()


def sort_compounds_by_frequency(compounds):
    """
    Sorts compounds by how frequent they are using the wordfreq library.

    :param compounds: row of a compound in a Pandas dataframe from the LaDEC dataset.
    :return: dictionary mapping each compound (constituent word 1, constituent word 2, plurality flag) to the frequency
    computed by wordfreq.
    """
    compound_freq = {(compound["c1"], compound["c2"], compound["isPlural"]): word_frequency(compound["stim"], "en")
                     for _, compound in compounds.iterrows()}
    compound_freq = dict(sorted(compound_freq.items(), key=lambda x: x[1], reverse=True))
    compounds = list(compound_freq.keys())
    return compounds


def sort_phrases_by_difficulty(phrases):
    """
    Sort input phrases by difficulty (see the compute_difficulty function in RebusGraph.py).

    :param phrases: list of phrases.
    :return: a dictionary mapping each phrase to its difficulty (sorted in descending order).
    """
    difficulty_freq = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    idiom_to_difficulty = {}
    for idiom in tqdm(phrases, desc="Computing difficulty (phrases)"):
        graphs = phrase_parser.parse(idiom)
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
    return idiom_to_difficulty


# Load compounds and sort them by frequency
compounds = pd.read_csv("../data/input/ladec_raw_small.csv")
compounds = sort_compounds_by_frequency(compounds)

# Load phrases and sort them by difficulty
with open("../data/input/idioms_raw.json", "r") as file:
    phrases = json.load(file)
    phrases = sort_phrases_by_difficulty(phrases)
    phrases = list(phrases.keys())


def generate_compounds():
    """
    Generates all puzzles from the list of compound words.
    """
    for c1, c2, is_plural in tqdm(compounds, desc="Generating puzzles (compounds)"):
        graphs = compound_parser.parse(c1, c2, is_plural)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"../results/benchmark/recent/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"../results/benchmark/recent/{save}")
            except:
                continue


def generate_phrases():
    """
    Generates all puzzles from the list of phrases.
    """
    for phrase in tqdm(phrases, desc="Generating puzzles (phrases)"):
        graphs = phrase_parser.parse(phrase)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"../results/benchmark/recent/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"../results/benchmark/recent/{save}")
            except:
                continue


def generate_custom_puzzles():
    """
    Generate puzzles from custom compounds and phrases.
    """
    with open(f"{os.path.dirname(__file__)}/data/input/custom_phrases.json", "r") as file:
        custom_phrases = json.load(file)
    custom_compounds = pd.read_csv(f"{os.path.dirname(__file__)}/data/input/custom_compounds.csv")

    for phrase in custom_phrases:
        graphs = phrase_parser.parse(phrase)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"../results/benchmark/recent/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"../results/benchmark/recent/{save}")
            except:
                continue

    for _, compound in custom_compounds.iterrows():
        c1, c2, is_plural = compound["c1"], compound["c2"], bool(compound["isPlural"])
        graphs = compound_parser.parse(c1, c2, is_plural)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"./results/benchmark/recent/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"./results/benchmark/recent/{save}")
            except:
                continue


def is_interesting(graph):
    """
    Checks automatically if a graph is 'interesting' (i.e., is not only text being placed next to each other).

    :param graph: rebus graph of a puzzle.
    :return: boolean indicating if a graph is 'interesting'.
    """
    edges = nx.get_edge_attributes(graph, "rule").values()
    if "INSIDE" in edges or "ABOVE" in edges or "OUTSIDE" in edges:
        return True
    node_attrs = get_node_attributes(graph)
    for node, attrs in node_attrs.items():
        if len(attrs) > 2 or (len(attrs) == 2 and attrs["repeat"] > 1):
            return True
    return False


def check_for_duplicates():
    """
    Prints any duplicate graphs in the final benchmark
    """
    phrase_graphs, compound_graphs = get_answer_graph_pairs("v3")
    for phrase_1, graph_1 in phrase_graphs.items():
        for phrase_2, graph_2 in phrase_graphs.items():
            if phrase_1 != phrase_2 and nx.utils.graphs_equal(graph_1, graph_2):
                print(phrase_1, phrase_2)
    for compound_1, graph_1 in phrase_graphs.items():
        for compound_2, graph_2 in phrase_graphs.items():
            if compound_1 != compound_2 and nx.utils.graphs_equal(graph_1, graph_2):
                print(compound_1, compound_2)


def generate_benchmark_file():
    """
    Generates the final JSON file for the benchmark, containing information on each puzzle (image, options, correct
    answer, if it is an icon puzzle, and if it is an overlap puzzle).
    """
    graphs = get_answer_graph_pairs(combine=True)
    puzzles = Benchmark().get_puzzles()
    all_puzzle_names = [os.path.basename(puzzle["image"]).split(".")[0] for puzzle in puzzles]
    benchmark = []
    for puzzle in puzzles:
        image = puzzle["image"]
        puzzle_name = os.path.basename(image).split(".")[0]
        options = puzzle["options"]
        correct = puzzle["correct"]

        is_icon = False
        graph = graphs[puzzle_name]
        contains_icons = sum([1 if "icon" in attr else 0 for attr in get_node_attributes(graph).values()]) > 0
        if contains_icons:
            is_icon = True

        is_overlap = False
        if puzzle_name.endswith("non-icon") or puzzle_name.endswith("icon"):
            text_puzzle_name = "_".join(puzzle_name.split("_")[:-1]) + "_non-icon"
            icon_puzzle_name = "_".join(puzzle_name.split("_")[:-1]) + "_icon"
            if text_puzzle_name in all_puzzle_names and icon_puzzle_name in all_puzzle_names:
                is_overlap = True

        benchmark.append({
            "image": os.path.basename(image),
            "options": options,
            "correct": correct,
            "is_icon": is_icon,
            "is_overlap": is_overlap
        })

    with open("../benchmark.json", "w") as file:
        json.dump(benchmark, file, indent=3)
