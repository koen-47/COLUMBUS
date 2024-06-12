import os
import json
import glob
import random

import pandas as pd

from util import get_answer_graph_pairs

random.seed(42)


class Benchmark:
    def __init__(self, with_metadata=True):
        images_dir = f"{os.path.dirname(__file__)}/images/*"
        distractors_path = f"{os.path.dirname(__file__)}/distractors/distractors.json"
        with open(distractors_path, "r") as file:
            distractors = json.load(file)
            images = {file: os.path.basename(file).split(".")[0] for file in glob.glob(images_dir)}
        self._puzzles = self._format_questions(images, distractors)

        if with_metadata:
            phrase_to_graph, compound_to_graph = get_answer_graph_pairs()
            all_to_graphs = phrase_to_graph.copy()
            all_to_graphs.update(compound_to_graph)

            for puzzle, graph in all_to_graphs.items():
                graph.graph = {}
                metadata = "\n".join(graph.__str__().split("\n")[1:])
                self._puzzles[puzzle]["metadata"] = {
                    "nodes": "\n".join([element for element in metadata.split("\n") if "rule:" not in element]),
                    "nodes_and_edges": metadata,
                }

    def get_puzzles(self):
        return list(self._puzzles.values())

    def _format_questions(self, images, distractors):
        questions = {file: {
            "options": distractors[file_base] + [" ".join(file_base.split("_")[:-1]) if file_base.split("_")[-1].isnumeric() else " ".join(file_base.split("_"))],
            "answer": " ".join(file_base.split("_")[:-1]) if file_base.split("_")[-1].isnumeric() else " ".join(file_base.split("_"))}
            for file, file_base in images.items()
            if file_base in distractors
        }

        questions = {file: {
            "options": random.sample(answers["options"], len(answers["options"])),
            "answer": answers["answer"]}
            for file, answers in questions.items()
        }

        questions = {file: {
            "options": {letter: option for letter, option in zip(["A", "B", "C", "D"], question["options"])},
            "correct": {["A", "B", "C", "D"][question["options"].index(question["answer"])]: question["answer"]}}
            for file, question in questions.items()
        }

        for file, question in questions.items():
            question["image"] = file
        questions = {os.path.basename(file).split(".")[0]: question for file, question in questions.items()}

        return questions
