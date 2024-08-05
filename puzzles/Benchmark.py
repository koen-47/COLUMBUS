import os
import json
import glob
import random

import pandas as pd

from util import get_answer_graph_pairs

random.seed(42)


class Benchmark:
    def __init__(self, with_metadata=True):
        images_dir = f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/*"
        distractors_path = f"{os.path.dirname(__file__)}/../saved/distractors_v3.json"
        with open(distractors_path, "r") as file:
            distractors = json.load(file)
            images = {file: os.path.basename(file).split(".")[0] for file in glob.glob(images_dir)}
        self._puzzles = self._format_questions(images, distractors)

        if with_metadata:
            graphs = get_answer_graph_pairs("v3", combine=True)
            for puzzle, graph in graphs.items():
                graph.graph = {}
                metadata = "\n".join(graph.__str__().split("\n")[1:])
                self._puzzles[puzzle]["metadata"] = {
                    "nodes": "\n".join([element for element in metadata.split("\n") if "rule:" not in element]),
                    "nodes_and_edges": metadata,
                }

    def get_puzzles(self):
        return list(self._puzzles.values())

    def _format_questions(self, images, distractors):
        questions = {}
        for file, file_base in images.items():
            options = distractors[file_base]
            if file_base.endswith("_icon") or file_base.endswith("_non-icon"):
                file_base = "_".join(file_base.split("_")[:-1])
            answer = " ".join(file_base.split("_")[:-1]) if file_base.split("_")[-1].isnumeric() else " ".join(file_base.split("_"))
            questions[file] = {"options": options + [answer], "answer": answer}

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
