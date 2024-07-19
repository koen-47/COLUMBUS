import json
import os

import networkx as nx
import pandas as pd
from tqdm import tqdm
from wordfreq import word_frequency

import util
from puzzles.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from puzzles.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from puzzles.RebusImageConverter import RebusImageConverter
from util import get_node_attributes, get_answer_graph_pairs

phrase_parser = PhraseRebusGraphParser()
compound_parser = CompoundRebusGraphParser()
generator = RebusImageConverter()


def sort_compounds_by_frequency(compounds):
    compound_freq = {(compound["c1"], compound["c2"], compound["isPlural"]): word_frequency(compound["stim"], "en")
                     for _, compound in compounds.iterrows()}
    compound_freq = dict(sorted(compound_freq.items(), key=lambda x: x[1], reverse=True))
    compounds = list(compound_freq.keys())
    return compounds


def sort_phrases_by_difficulty(phrases):
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


# compounds = pd.read_csv("./saved/ladec_raw_small.csv")
# compounds = sort_compounds_by_frequency(compounds)

# with open("./saved/idioms_raw.json", "r") as file:
#     phrases = json.load(file)
#     phrases = sort_phrases_by_difficulty(json.load(file))
#     phrases = list(phrases.keys())


def generate_compounds():
    # compounds_ = [(compound["c1"], compound["c2"], bool(compound["isPlural"])) for _, compound in compounds.iterrows()]
    for c1, c2, is_plural in tqdm(compounds, desc="Generating puzzles (compounds)"):
        graphs = compound_parser.parse(c1, c2, is_plural)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"./results/compounds/v5/compounds/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"./results/compounds/v5/compounds/{save}")
            except:
                continue


def generate_phrases():
    for phrase in tqdm(phrases, desc="Generating puzzles (phrases)"):
        graphs = phrase_parser.parse(phrase)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"./results/benchmark/phrases/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"./results/benchmark/phrases/{save}")
            except:
                continue


def generate_custom_puzzles():
    with open(f"{os.path.dirname(__file__)}/saved/custom_phrases.json", "r") as file:
        custom_phrases = json.load(file)
    custom_compounds = pd.read_csv(f"{os.path.dirname(__file__)}/saved/custom_compounds.csv")

    for phrase in custom_phrases:
        graphs = phrase_parser.parse(phrase)
        if graphs is not None:
            try:
                if len(graphs) == 1:
                    graph = graphs[0]
                    if is_interesting(graph):
                        save = "_".join(graph.graph["answer"].lower().split())
                        generator.generate(graph, show=False, save=f"./results/benchmark/custom/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"./results/benchmark/custom/{save}")
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
                        generator.generate(graph, show=False, save=f"./results/benchmark/custom/{save}")
                else:
                    for i, graph in enumerate(graphs):
                        if is_interesting(graph):
                            save = "_".join(graph.graph["answer"].lower().split()) + f"_{i + 1}"
                            generator.generate(graph, show=False, save=f"./results/benchmark/custom/{save}")
            except:
                continue


def is_interesting(graph):
    edges = nx.get_edge_attributes(graph, "rule").values()
    if "INSIDE" in edges or "ABOVE" in edges or "OUTSIDE" in edges:
        return True
    node_attrs = get_node_attributes(graph)
    for node, attrs in node_attrs.items():
        if len(attrs) > 2 or (len(attrs) == 2 and attrs["repeat"] > 1):
            return True
    return False


def check_for_duplicates():
    phrase_graphs, compound_graphs = get_answer_graph_pairs()
    for phrase_1, graph_1 in phrase_graphs.items():
        for phrase_2, graph_2 in phrase_graphs.items():
            if phrase_1 != phrase_2 and nx.utils.graphs_equal(graph_1, graph_2):
                print(phrase_1, phrase_2)
    for compound_1, graph_1 in phrase_graphs.items():
        for compound_2, graph_2 in phrase_graphs.items():
            if compound_1 != compound_2 and nx.utils.graphs_equal(graph_1, graph_2):
                print(compound_1, compound_2)


# generate_phrases()
# generate_compounds()
# generate_custom_puzzles()
# check_for_duplicates()
