import json
import os
import re
from itertools import product

import wordfreq
import matplotlib.pyplot as plt


class AnalysisReport:
    def __init__(self):
        self.results_dir = f"{os.path.dirname(__file__)}/results"

    def generate_all(self):
        with open(f"{self.results_dir}/clip_compounds.json", "r") as file:
            results = json.load(file)
            correct_answers = [list(result["correct"].keys())[0] for result in results]
            correct_freq = {}
            for answer in correct_answers:
                if answer not in correct_freq:
                    correct_freq[answer] = 0
                correct_freq[answer] += 1

            correct_freq = {answer: freq/sum(correct_freq.values()) for answer, freq in correct_freq.items()}
            plt.pie(list(correct_freq.values()), labels=list(correct_freq.keys()), autopct='%1.1f%%', startangle=90)
            plt.show()

        self.generate("compounds", "clip", prompt_type="")
        self.generate("phrases", "clip", prompt_type="")

        puzzle_types = ["compounds", "phrases"]
        model_types = ["blip-2_opt-2.7b", "blip-2_opt-6.7b", "blip-2_flan-t5-xxl",
                       "fuyu-8b", "instructblip", "llava-1.5-13b"]
        prompt_types = ["1", "2"]

        for puzzle, model, prompt in product(*[puzzle_types, model_types, prompt_types]):
            self.generate(puzzle, model, prompt)

    def generate(self, puzzle_type, model_type, prompt_type, mistral_type=None):
        if model_type == "clip":
            with open(f"{self.results_dir}/{model_type}_{puzzle_type}.json", "r") as file:
                results = json.load(file)
        elif model_type == "mistral-7b":
            with open(f"{self.results_dir}/{model_type}_{puzzle_type}.json", "r") as file:
                results = json.load(file)["results"]
        else:
            with open(f"{self.results_dir}/prompt_{prompt_type}/{model_type}_{puzzle_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)
                if prompt_type == "2":
                    results = results["results"]

        counter = 0
        for result in results:
            if model_type == "llava-1.5-13b":
                result = self._preprocess_llava_result(result)
            if model_type == "fuyu-8b":
                result, counter = self._preprocess_fuyu_result(result, counter)
            elif model_type == "mistral-7b":
                is_phrase = puzzle_type == "phrases"
                result, counter = self._preprocess_mistral_result(result, counter, is_phrase=is_phrase)
            result = self._standardize_general_result(result, mistral_type=mistral_type)

        print(f"\n=== ANALYSIS {model_type.upper()} ===")
        print(f"Puzzle type: {puzzle_type}")
        print(f"Prompt includes description: {prompt_type == '2'}")
        self.analyze_basic_models(results, puzzle_type=puzzle_type)


    def analyze_basic_models(self, results, puzzle_type):
        def compute_accuracy(results):
            n_correct = len([result for result in results if result["is_correct"] is True])
            return (n_correct / len(results)) * 100

        def compute_max_answer_occurrence(results):
            answers = [list(result["clean_output"].keys())[0] for result in results if "clean_output" in result]
            answers_freq = {}
            for answer in answers:
                if answer not in answers_freq:
                    answers_freq[answer] = 0
                answers_freq[answer] += 1

            answers_freq = {answer: freq/sum(answers_freq.values()) for answer, freq in answers_freq.items()}
            return {max(answers_freq, key=answers_freq.get): round(answers_freq[max(answers_freq, key=answers_freq.get)] * 100, 2)}

        accuracy = compute_accuracy(results)
        max_answer_occurrence = compute_max_answer_occurrence(results)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Max answer occurrence proportion: {max_answer_occurrence}")

        if puzzle_type == "compounds":
            compound_freqs = {}
            for result in results:
                correct = list(result["correct"].values())[0]
                compound_freqs[correct] = wordfreq.word_frequency(correct, "en", wordlist="best", minimum=0.0)
            compound_freqs = dict(sorted(compound_freqs.items(), key=lambda item: item[1], reverse=True))
            compound_freq_top50 = list(compound_freqs.keys())[:int(len(compound_freqs) * 0.5)]
            compound_freq_top10 = list(compound_freqs.keys())[:int(len(compound_freqs) * 0.1)]
            results_top50 = [result for result in results if list(result["correct"].values())[0] in compound_freq_top50]
            results_top10 = [result for result in results if list(result["correct"].values())[0] in compound_freq_top10]
            print(f"Top 50% accuracy: {compute_accuracy(results_top50):.2f}")
            print(f"Top 10% accuracy: {compute_accuracy(results_top10):.2f}")



    def _standardize_general_result(self, result, mistral_type=None):
        output = result["output"]
        if mistral_type is not None:
            output = result["output"][mistral_type]
        if re.match(r"\([A-D]\)\s.+", output):
            letter = output.split()[0][1]
            answer = " ".join(output.split()[1:])
            is_correct = {letter: answer} == result["correct"]
            result["is_correct"] = is_correct
            result["clean_output"] = {letter: answer}
        elif re.match(r"[A-D]\)\s.+", output):
            letter = output.split()[0][0]
            answer = " ".join(output.split()[1:])
            is_correct = {letter: answer} == result["correct"]
            result["is_correct"] = is_correct
            result["clean_output"] = {letter: answer}
        elif re.match(r"^[A-D]$", output):
            letter = output
            answer = result["options"][letter]
            is_correct = {letter: answer} == result["correct"]
            result["is_correct"] = is_correct
            result["clean_output"] = {letter: answer}
        else:
            answer_to_letter = {v: k for k, v in result["options"].items()}
            if output in answer_to_letter:
                letter = answer_to_letter[output]
                is_correct = {letter: output} == result["correct"]
                result["is_correct"] = is_correct
                result["clean_output"] = {letter: output}
            else:
                result["is_correct"] = False
        return result

    def _preprocess_llava_result(self, result):
        result["output"] = re.split("ASSISTANT: ", result["output"])[1]
        return result

    def _preprocess_fuyu_result(self, result, counter):
        output = result["output"]
        if re.match(r"^\) [A-z ]*\n", output):
            result["output"] = " ".join(output.split("\n")[0].split()[1:])
            counter += 1
        elif re.match(r"[A-z -]*\n", output):
            result["output"] = output.split("\n")[0]
            counter += 1
        elif re.match(r"^([A-D]\)) [A-z ]*\n", output):
            result["output"] = output.split("\n")[0]
            counter += 1
        elif re.match(r"^\([A-D]\) [A-z ]*\n", output):
            result["output"] = output.split("\n")[0]
            counter += 1
        elif re.match(r"^(\u0004) [A-z ]*", output):
            result["output"] = " ".join(output.split("\n")[0].split()[1:])
            counter += 1
        return result, counter

    def _preprocess_mistral_result(self, result, counter, is_phrase=False):
        if is_phrase:
            output_nodes_edges = result["output"]["nodes_and_edges"]
            output_nodes = result["output"]["nodes"]
            match_nodes_edges = re.search(r"\(\((.*?)\)\)", output_nodes_edges)
            match_nodes = re.search(r"\(\((.*?)\)\)", output_nodes)
            if match_nodes_edges:
                result["output"]["nodes_and_edges"] = match_nodes_edges.group(1)
                counter += 1
            if match_nodes:
                result["output"]["nodes"] = match_nodes.group(1)
                counter += 1
            return result, counter

        output = result["output"]
        match = re.search(r"\(\((.*?)\)\)", output)
        if match:
            result["output"] = match.group(1)
            counter += 1
            return result, counter
        return result, counter

