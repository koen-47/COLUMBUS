import json
import os
import copy
import glob

from tqdm import tqdm
import networkx as nx

from util import get_answer_graph_pairs, get_node_attributes
from puzzles.RebusImageConverter import RebusImageConverter


def switch_icons():
    image_generator = RebusImageConverter()
    graphs = get_answer_graph_pairs("v2", combine=True)
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
            image_generator.generate(graph, save=f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/{answer}_icon.png")
            image_generator.generate(graph_no_icon, save=f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/{answer}_non-icon.png")
            n_icon_puzzles += 1
            n_non_icon_puzzles += 1
        else:
            image_generator.generate(graph_no_icon, save=f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/{answer}.png")
            n_non_icon_puzzles += 1

    print(n_non_icon_puzzles, n_icon_puzzles)


def rename_distractors():
    with open(f"{os.path.dirname(__file__)}/../saved/distractors.json", "r") as file:
        distractors = json.load(file)
    phrases = [os.path.basename(file).split(".")[0] for file in glob.glob(f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/*")]
    phrases_to_remove = []
    for phrase in phrases:
        if phrase.endswith("_icon") or phrase.endswith("_non-icon"):
            phrase_ = "_".join(phrase.split("_")[:-1])
            distractors[phrase] = distractors[phrase_]
            phrases_to_remove.append(phrase_)
    phrases_to_remove = set(phrases_to_remove)
    distractors = {answer: distractors_ for answer, distractors_ in distractors.items() if answer not in phrases_to_remove
                   and answer not in set(distractors.keys()).difference(phrases)}

    with open(f"{os.path.dirname(__file__)}/../saved/distractors_v3.json", "w") as file:
        json.dump(distractors, file, indent=3)

    # print(json.dumps(distractors, indent=3))
    print(len(phrases))
    print(len(distractors))
    # print(set(phrases).difference(distractors.keys()))


# switch_icons()
rename_distractors()

# def analyze_switched_icon_puzzles():
#     puzzles = get_answer_graph_pairs(combine=True)
#     n_non_overlap_puzzles, n_icon_overlap_puzzles = 0, 0,
#     for file in glob.glob(f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/*"):
#         file_name = os.path.basename(file).split(".")[0]
#         if file_name.endswith("icon") or file_name.endswith("non-icon"):
#             n_icon_overlap_puzzles += 1
#         else:
#             n_non_overlap_puzzles += 1
#
#     print(n_non_overlap_puzzles)
#     print(n_icon_overlap_puzzles)
#
#
# analyze_switched_icon_puzzles()
