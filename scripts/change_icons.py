"""
Scripts used to convert icons in a puzzle to their textual counterpart. NOTE: these scripts no longer work, but are
still worth including for reproducibility and documentation.
"""

import json
import os
import copy
import glob

from tqdm import tqdm
import networkx as nx

from util import get_answer_graph_pairs, get_node_attributes
from puzzles.RebusImageConverter import RebusImageConverter


def switch_icons():
    """
    Switches the icons in all puzzles to their textual counterpart.
    """
    image_generator = RebusImageConverter()
    graphs = get_answer_graph_pairs(combine=True)
    n_icon_puzzles, n_non_icon_puzzles, n_overlap_puzzles = 0, 0, 0
    for answer, graph in tqdm(graphs.items(), desc="Switching icons"):
        graph_no_icon = copy.deepcopy(graph)
        switched_icon = False
        graph_no_icon_node_attrs = get_node_attributes(graph_no_icon)
        for attr in graph_no_icon_node_attrs.values():
            if "icon" in attr:
                attr["text"] = list(attr["icon"].keys())[0].upper()
                del attr["icon"]
                switched_icon = True

        for node in graph_no_icon.nodes:
            graph_no_icon.nodes[node].clear()
        nx.set_node_attributes(graph_no_icon, graph_no_icon_node_attrs)

        if switched_icon:
            image_generator.generate(graph, save=f"{os.path.dirname(__file__)}/../results/benchmark/images/{answer}_icon.png")
            image_generator.generate(graph_no_icon, save=f"{os.path.dirname(__file__)}/../results/benchmark/images/{answer}_non-icon.png")
            n_icon_puzzles += 1
            n_non_icon_puzzles += 1
        else:
            image_generator.generate(graph_no_icon, save=f"{os.path.dirname(__file__)}/../results/benchmark/images/{answer}.png")
            n_non_icon_puzzles += 1


def rename_distractors():
    """
    Renames all distractors of the pre-icon switched puzzles to their new name.
    """
    with open(f"{os.path.dirname(__file__)}/../distractors/distractors_v3.json", "r") as file:
        distractors = json.load(file)
    phrases = [os.path.basename(file).split(".")[0] for file in glob.glob(f"{os.path.dirname(__file__)}/../results/"
                                                                          f"benchmark/images/*")]
    phrases_to_remove = []
    for phrase in phrases:
        if phrase.endswith("_icon") or phrase.endswith("_non-icon"):
            phrase_ = "_".join(phrase.split("_")[:-1])
            distractors[phrase] = distractors[phrase_]
            phrases_to_remove.append(phrase_)
    phrases_to_remove = set(phrases_to_remove)
    distractors = {answer: distractors_ for answer, distractors_ in distractors.items() if answer not in phrases_to_remove
                   and answer not in set(distractors.keys()).difference(phrases)}

    with open(f"{os.path.dirname(__file__)}/../distractors/distractors_v3.json", "w") as file:
        json.dump(distractors, file, indent=3)


def analyze_switched_icon_puzzles():
    """
    Performs some basic analysis of the new puzzles after they have their icons switched.
    """
    puzzles = get_answer_graph_pairs(combine=True)
    n_non_icon_puzzles, n_icon_puzzles, n_icon_overlap_puzzles = 0, 0, 0
    for answer, graph in puzzles.items():
        node_attrs = get_node_attributes(graph)
        contains_icons = sum([1 if "icon" in attr else 0 for attr in node_attrs.values()]) > 0
        if contains_icons:
            n_icon_puzzles += 1
        else:
            n_non_icon_puzzles += 1

        if answer.endswith("icon") or answer.endswith("non-icon"):
            n_icon_overlap_puzzles += 1

    print("Number of non-icon puzzles:", n_non_icon_puzzles)
    print("Number of icon puzzles:", n_icon_puzzles)
    print(f"Number of puzzles with icon and non-icon variant:", n_icon_overlap_puzzles)
    print(f"Total number of puzzles:", len(puzzles))
