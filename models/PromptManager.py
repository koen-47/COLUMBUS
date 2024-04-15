import os
import json
import glob
import random

random.seed(42)


class PromptManager:
    def __init__(self):
        self.template = "Question: Which word/phrase best describes this image?\n" \
                        "(A) {}\n" \
                        "(B) {}\n" \
                        "(C) {}\n" \
                        "(D) {}\n" \
                        "Answer:"

        with open(f"{os.path.dirname(__file__)}/../saved/compound_distractors_final.json", "r") as file:
            self.compound_distractors = json.load(file)
            self.compound_images = {file: os.path.basename(file).split(".")[0].split("_")[0] for file in
                                    glob.glob(f"{os.path.dirname(__file__)}/../results/compounds/saved/*")}
            self.compounds = {file: {"options": self.compound_distractors[file_base] + [file_base], "answer": file_base}
                              for file, file_base in self.compound_images.items() if
                              file_base in self.compound_distractors}
            self.compounds = {file: {"options": random.sample(answers["options"], len(answers["options"])),
                                     "answer": answers["answer"]} for file, answers in self.compounds.items()}

        with open(f"{os.path.dirname(__file__)}/../saved/idiom_distractors_final.json", "r") as file:
            self.phrase_distractors = json.load(file)
            self.phrase_images = {file: " ".join(os.path.basename(file).split(".")[0].split("_")) for file in
                                  glob.glob(f"{os.path.dirname(__file__)}/../results/idioms/all/*")}
            self.phrases = {file: {"options": self.phrase_distractors[file_base] + [file_base], "answer": file_base} for
                            file, file_base in self.phrase_images.items() if file_base in self.phrase_distractors}
            self.phrases = {file: {"options": random.sample(answers["options"], len(answers["options"])),
                                   "answer": answers["answer"]} for file, answers in self.phrases.items()}

        self.compound_image_question_pairs = []
        for image, question in self.compounds.items():
            options = [option.capitalize() for option in question["options"]]
            prompt = self.template.format(*options)
            self.compound_image_question_pairs.append({"image": image, "question": prompt, "correct": question["answer"].capitalize()})

        self.phrase_image_question_pairs = []
        for image, question in self.phrases.items():
            options = [option.capitalize() for option in question["options"]]
            prompt = self.template.format(*options)
            self.phrase_image_question_pairs.append({"image": image, "question": prompt, "correct": question["answer"].capitalize()})
