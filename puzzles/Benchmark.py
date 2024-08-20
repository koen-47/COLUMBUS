import os
import json
import glob
import random

from util import get_answer_graph_pairs

# Set seed to 42
random.seed(42)


class Benchmark:
    """
    Class to hold the benchmark
    """
    def __init__(self, with_metadata=True):
        # Load images of puzzles and distractors
        images_dir = f"{os.path.dirname(__file__)}/../results/benchmark/images/*"
        distractors_path = f"{os.path.dirname(__file__)}/../data/distractors/distractors_v3.json"
        with open(distractors_path, "r") as file:
            distractors = json.load(file)
            images = {file: os.path.basename(file).split(".")[0] for file in glob.glob(images_dir)}

        # Format the puzzles
        self._puzzles = self._format_questions(images, distractors)

        # Add the metadata of the graph descriptions (if specified)
        if with_metadata:
            graphs = get_answer_graph_pairs(combine=True)
            for puzzle, graph in graphs.items():
                graph.graph = {}
                metadata = "\n".join(graph.__str__().split("\n")[1:])
                self._puzzles[puzzle]["metadata"] = {
                    "nodes": "\n".join([element for element in metadata.split("\n") if "rule:" not in element]),
                    "nodes_and_edges": metadata,
                }

    def get_puzzles(self):
        """
        Returns the puzzles as a list.
        :return: list of puzzles
        """
        return list(self._puzzles.values())

    def _format_questions(self, images, distractors):
        """
        Format each question in the form {options: {A: ..., B: ..., C: ..., D: ...}, correct: ...}
        :param images: list of image paths to each puzzle.
        :param distractors: dictionary that maps a puzzle's correct answer to its distractors.
        :return: a list of dictionaries containing the image, options and correct answer for each puzzle.
        """

        # Create dictionary that maps each puzzle to its options and correct answer
        questions = {}
        for file, file_base in images.items():
            options = distractors[file_base]
            if file_base.endswith("_icon") or file_base.endswith("_non-icon"):
                file_base = "_".join(file_base.split("_")[:-1])
            answer = " ".join(file_base.split("_")[:-1]) if file_base.split("_")[-1].isnumeric() else " ".join(file_base.split("_"))
            questions[file] = {"options": options + [answer], "answer": answer}

        # Shuffle the options of each question
        questions = {file: {
            "options": random.sample(answers["options"], len(answers["options"])),
            "answer": answers["answer"]}
            for file, answers in questions.items()
        }

        # Map each option to A, B, C, or D
        questions = {file: {
            "options": {letter: option for letter, option in zip(["A", "B", "C", "D"], question["options"])},
            "correct": {["A", "B", "C", "D"][question["options"].index(question["answer"])]: question["answer"]}}
            for file, question in questions.items()
        }

        # Add the image path to each question
        for file, question in questions.items():
            question["image"] = file
        questions = {os.path.basename(file).split(".")[0]: question for file, question in questions.items()}

        return questions
