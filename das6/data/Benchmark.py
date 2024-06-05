import os
import json
import glob
import random

import pandas as pd

from parsers.RebusGraphParser import RebusGraphParser

random.seed(42)


class Benchmark:
    def __init__(self, with_metadata=True):
        compound_images_dir = f"{os.path.dirname(__file__)}/images/compounds/*"
        compound_distractors_path = f"{os.path.dirname(__file__)}/distractors/compound_distractors_final.json"
        with open(compound_distractors_path, "r") as file:
            compound_distractors = json.load(file)
            compound_images = {file: os.path.basename(file).split(".")[0].split("_")[0] for file in glob.glob(compound_images_dir)}
        self._compounds = self._format_questions(compound_images, compound_distractors)

        phrases_images_dir = f"{os.path.dirname(__file__)}/images/phrases/*"
        phrases_distractors_path = f"{os.path.dirname(__file__)}/distractors/idiom_distractors_final.json"
        with open(phrases_distractors_path, "r") as file:
            phrase_distractors = json.load(file)
            phrase_images = {file: " ".join(os.path.basename(file).split(".")[0].split("_")) for file in
                             glob.glob(phrases_images_dir)}
        self._phrases = self._format_questions(phrase_images, phrase_distractors)

        if with_metadata:
            parser = RebusGraphParser(f"{os.path.dirname(__file__)}/misc/ladec_raw_small.csv")
            for compound in self._compounds:
                correct, image = list(compound["correct"].values())[0], compound["image"]
                graphs = parser.parse_compound(compound=correct)
                if len(graphs) > 1:
                    graph = graphs[int(os.path.basename(image).split(".")[0].split("_")[1])-1]
                else:
                    graph = graphs[0]
                graph.graph = {}
                metadata = "\n".join(graph.__str__().split("\n")[1:])
                compound["metadata"] = metadata

            for phrase in self._phrases:
                correct, image = list(phrase["correct"].values())[0], phrase["image"]
                graph = parser.parse_idiom(correct)
                graph.graph = {}
                metadata = "\n".join(graph.__str__().split("\n")[1:])
                phrase["metadata"] = {
                    "nodes_and_edges": metadata,
                    "nodes": "\n".join([element for element in metadata.split("\n") if "rule:" not in element])
                }

    def get_puzzles(self):
        return self._compounds, self._phrases

    def _format_questions(self, images, distractors):
        questions = {file: {"options": distractors[file_base] + [file_base], "answer": file_base} for file, file_base in images.items() if file_base in distractors}
        questions = {file: {"options": random.sample(answers["options"], len(answers["options"])), "answer": answers["answer"]} for file, answers in questions.items()}
        questions = {file: {"options": {letter: option for letter, option in zip(["A", "B", "C", "D"], question["options"])}, "correct": {["A", "B", "C", "D"][question["options"].index(question["answer"])]: question["answer"]}} for file, question in questions.items()}

        image_questions = []
        for file, question in questions.items():
            question["image"] = file
            image_questions.append(question)

        return image_questions
