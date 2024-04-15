import os
import json
import glob
import random

random.seed(42)


class PromptManager:
    def __init__(self):
        compound_images_dir = f"{os.path.dirname(__file__)}/../results/compounds/saved/*"
        compound_distractors_path = f"{os.path.dirname(__file__)}/../saved/compound_distractors_final.json"
        with open(compound_distractors_path, "r") as file:
            compound_distractors = json.load(file)
            compound_images = {file: os.path.basename(file).split(".")[0].split("_")[0] for file in glob.glob(compound_images_dir)}
        self.compounds = self._format_questions(compound_images, compound_distractors)

        phrases_images_dir = f"{os.path.dirname(__file__)}/../results/idioms/all/*"
        phrases_distractors_path = f"{os.path.dirname(__file__)}/../saved/idiom_distractors_final.json"
        with open(phrases_distractors_path, "r") as file:
            phrase_distractors = json.load(file)
            phrase_images = {file: " ".join(os.path.basename(file).split(".")[0].split("_")) for file in
                             glob.glob(phrases_images_dir)}
        self.phrases = self._format_questions(phrase_images, phrase_distractors)

    def _format_questions(self, images, distractors):
        questions = {file: {"options": distractors[file_base] + [file_base], "answer": file_base} for file, file_base in images.items() if file_base in distractors}
        questions = {file: {"options": random.sample(answers["options"], len(answers["options"])), "answer": answers["answer"]} for file, answers in questions.items()}
        questions = {file: {"options": {letter: option for letter, option in zip(["A", "B", "C", "D"], question["options"])}, "correct": {["A", "B", "C", "D"][question["options"].index(question["answer"])]: question["answer"]}} for file, question in questions.items()}

        image_questions = []
        for file, question in questions.items():
            question["image"] = file
            image_questions.append(question)

        return image_questions
