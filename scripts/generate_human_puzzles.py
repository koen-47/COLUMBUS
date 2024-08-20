"""
Script to generate the 105 puzzle subset used for evaluating human performance.
"""

import json
import random

from util import get_answer_graph_pairs, get_node_attributes

random.seed(42)


with open("../benchmark.json", "r") as file:
    benchmark = json.load(file)
    puzzles = list(get_answer_graph_pairs("v3").items())
    random.shuffle(puzzles)
    puzzles = dict(puzzles)

    n_puzzles = int(len(benchmark) / 10)
    n_non_icon_puzzles, n_icon_puzzles = [40] * 2
    n_overlap_puzzles = 28
    icon_puzzles_sample, non_icon_puzzles_sample = [], []
    overlap_puzzles_sample = []

    for answer, graph in puzzles.items():
        contains_icon = sum([1 if "icon" in attr else 0 for attr in get_node_attributes(graph).values()]) > 0
        if contains_icon and len(icon_puzzles_sample) < n_icon_puzzles:
            icon_puzzles_sample.append(answer)
        elif not contains_icon and len(non_icon_puzzles_sample) < n_non_icon_puzzles:
            non_icon_puzzles_sample.append(answer)

        if len(non_icon_puzzles_sample) == n_non_icon_puzzles and len(icon_puzzles_sample) == n_icon_puzzles:
            break

    for answer, graph in reversed(puzzles.items()):
        non_icon_puzzle = "_".join(answer.split("_")[:-1]) + "_non-icon"
        icon_puzzle = "_".join(answer.split("_")[:-1]) + "_icon"

        if answer.endswith("icon") or answer.endswith("non-icon"):
            non_icon_puzzle = "_".join(answer.split("_")[:-1]) + "_non-icon"
            icon_puzzle = "_".join(answer.split("_")[:-1]) + "_icon"
            if non_icon_puzzle in puzzles.keys() and icon_puzzle in puzzles.keys():
                overlap_puzzles_sample.append(non_icon_puzzle)
                overlap_puzzles_sample.append(icon_puzzle)

        if len(overlap_puzzles_sample) == n_overlap_puzzles:
            break

    samples = list(set(non_icon_puzzles_sample + icon_puzzles_sample + overlap_puzzles_sample))
    benchmark_human = []

    for puzzle in benchmark:
        image = puzzle["image"].split(".")[0]
        if image in samples:
            benchmark_human.append(puzzle)

    random.shuffle(benchmark_human)

    print("Number of non-icon puzzles:", len(non_icon_puzzles_sample))
    print("Number of icon puzzles:", len(icon_puzzles_sample))
    print("Number of overlap puzzles:", len(overlap_puzzles_sample))
    print("Total number of puzzles:", len(samples))

    with open("../results/analysis/results/human/benchmark_human.json", "w") as file_:
        json.dump(benchmark_human, file_, indent=3)
