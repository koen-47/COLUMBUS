import glob
import json
import os
import re
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import wordfreq
import matplotlib.pyplot as plt
from tqdm import tqdm

from graphs.legacy.RebusGraphParser import RebusGraphParser
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.patterns.Rule import Rule
from util import get_node_attributes, get_answer_graph_pairs


class AnalysisReport:
    def __init__(self):
        self.results_dir = f"{os.path.dirname(__file__)}/results_v2"
        self._graph_answer_pairs = get_answer_graph_pairs(combine=True)

    def generate_all(self, verbose=False):
        model_types = ["blip-2_opt-2.7b", "blip-2_opt-6.7b", "blip-2_flan-t5-xxl",
                       "fuyu-8b", "instructblip", "llava-1.5-13b", "llava-1.6-34b", "cogvlm", "qwenvl", "mistral-7b"]
        prompt_types = ["1", "2", "3", "4"]
        all_basic_results = {prompt: {model: None} for model, prompt in product(*[model_types, prompt_types])}
        all_rule_results = {prompt: {model: None} for model, prompt in product(*[model_types, prompt_types])}

        all_basic_results["1"]["mistral-7b"] = ["-"] * 4
        all_basic_results["2"]["mistral-7b"] = ["-"] * 4
        all_basic_results["1"]["human"] = ["-"] * 4
        all_basic_results["3"]["human"] = ["-"] * 4
        all_basic_results["4"]["human"] = ["-"] * 4

        self.generate("clip", prompt_type="N/A")
        for model, prompt in product(*[model_types, prompt_types]):
            if model == "mistral-7b" and (prompt == "1" or prompt == "2"):
                continue

            if verbose:
                print(f"\n=== MODEL EVALUATION ANALYSIS ({model.upper()}) ===")
                print(f"Prompt type: {prompt}")

            basic_results, rule_results = self.generate(model, prompt, verbose=verbose)
            all_basic_results[prompt][model] = basic_results
            if model != "blip-2_opt-2.7b" and model != "blip-2_opt-6.7b" and model != "instructblip" and model != "mistral-7b":
                all_rule_results[prompt][model] = rule_results

        human_results = []
        n_icon_puzzles, n_non_icon_puzzles = 0, 0
        for file_path in glob.glob(f"{self.results_dir}/human/*"):
            with open(file_path, "r") as file:
                results = json.load(file)
            for result in results:
                self._standardize_general_result(result)
                img = os.path.basename(result["image"]).split(".")[0]
                node_attrs = get_node_attributes(self._graph_answer_pairs[img])
                contains_icons = sum([1 if "icon" in attr else 0 for attr in node_attrs.values()]) > 0
                if contains_icons:
                    n_icon_puzzles += 1
                else:
                    n_non_icon_puzzles += 1
            human_results = self.analyze_basic(results)
        all_basic_results["2"]["human"] = human_results

        # print(n_non_icon_puzzles, n_icon_puzzles)

        self.analyze_overall(all_basic_results, all_rule_results, verbose=True)

    def generate(self, model_type, prompt_type, mistral_type=None, verbose=False):
        if model_type == "clip":
            with open(f"{self.results_dir}/{model_type}.json", "r") as file:
                results = json.load(file)["results"]
        elif model_type == "mistral-7b":
            with open(f"{self.results_dir}/{model_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)["results"]
        else:
            with open(f"{self.results_dir}/prompt_{prompt_type}/{model_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)["results"]

        counter = 0
        for result in results:
            if model_type == "llava-1.5-13b":
                result = self._preprocess_llava_13b_result(result)
            if model_type == "llava-1.6-34b":
                result, counter = self._preprocess_llava_34b_result(result, counter)
            if model_type == "fuyu-8b":
                result, counter = self._preprocess_fuyu_result(result, counter)
            if model_type == "cogvlm":
                result, counter = self._preprocess_cogvlm_result(result, counter)
            if model_type == "qwenvl":
                result, counter = self._preprocess_qwenvl_result(result, counter)
            elif model_type == "mistral-7b":
                result, counter = self._preprocess_mistral_result(result, counter)
            result = self._standardize_general_result(result, mistral_type=mistral_type)

        basic_results = self.analyze_basic(results, verbose=verbose)
        rule_results = self.analyze_by_rule(results)
        return basic_results, rule_results

    def analyze_by_rule(self, results, verbose=False):
        rules = list(Rule.get_all_rules()["individual"].keys()) + ["sound"]
        rules_freq_text, rules_freq_icon = {}, {}
        edge_freq_text, edge_freq_icon = {}, {}

        def increment_rule_freq(rule, value, contains_icons):
            if contains_icons:
                if rule in rules:
                    if rule not in rules_freq_icon:
                        rules_freq_icon[rule] = []
                    if result["is_correct"]:
                        rules_freq_icon[rule].append(1)
                    else:
                        rules_freq_icon[rule].append(0)
                if f"{rule}_{value}" in rules:
                    if f"{rule}_{value}" not in rules_freq_icon:
                        rules_freq_icon[f"{rule}_{value}"] = []
                    if result["is_correct"]:
                        rules_freq_icon[f"{rule}_{value}"].append(1)
                    else:
                        rules_freq_icon[f"{rule}_{value}"].append(0)
            else:
                if rule in rules:
                    if rule not in rules_freq_text:
                        rules_freq_text[rule] = []
                    if result["is_correct"]:
                        rules_freq_text[rule].append(1)
                    else:
                        rules_freq_text[rule].append(0)
                if f"{rule}_{value}" in rules:
                    if f"{rule}_{value}" not in rules_freq_text:
                        rules_freq_text[f"{rule}_{value}"] = []
                    if result["is_correct"]:
                        rules_freq_text[f"{rule}_{value}"].append(1)
                    else:
                        rules_freq_text[f"{rule}_{value}"].append(0)

        def format_attrs(attrs):
            attrs_ = attrs.copy()
            del attrs_["text"]
            if attrs_["repeat"] == 1:
                del attrs_["repeat"]
            for rule, value in attrs_.copy().items():
                if value == 2:
                    attrs_[rule] = "two"
                if value == 4:
                    attrs_[rule] = "four"
                if rule == "repeat":
                    attrs_["repetition"] = attrs_[rule]
                    del attrs_[rule]
            return attrs_

        for result in results:
            graph_name = os.path.basename(result["image"]).split(".")[0]
            graph = self._graph_answer_pairs[graph_name]
            node_attrs = get_node_attributes(graph)
            contains_icons = sum([1 if "icon" in attr else 0 for attr in node_attrs.values()]) > 0
            for node, attrs in node_attrs.items():
                attrs_ = format_attrs(attrs)
                for rule, value in attrs_.items():
                    increment_rule_freq(rule, value, contains_icons)
            edges = nx.get_edge_attributes(graph, "rule").values()
            for edge in edges:
                if not contains_icons:
                    if edge not in edge_freq_text:
                        edge_freq_text[edge] = []
                    if result["is_correct"]:
                        edge_freq_text[edge].append(1)
                    else:
                        edge_freq_text[edge].append(0)
                else:
                    if edge not in edge_freq_icon:
                        edge_freq_icon[edge] = []
                    if result["is_correct"]:
                        edge_freq_icon[edge].append(1)
                    else:
                        edge_freq_icon[edge].append(0)

        for rule, freq in rules_freq_text.items():
            rules_freq_text[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))
        for rule, freq in edge_freq_text.items():
            edge_freq_text[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))
        for rule, freq in rules_freq_icon.items():
            rules_freq_icon[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))
        for rule, freq in edge_freq_icon.items():
            edge_freq_icon[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))

        if verbose:
            print("=== RULES ANALYSIS (NO ICONS) ===")
            print(f"Unique individual rules frequency:", len(rules_freq_text))
            print(json.dumps(rules_freq_text, indent=3))
            print(json.dumps(edge_freq_text, indent=3))

            print("\n=== RULES ANALYSIS (ICONS) ===")
            print(f"Unique individual rules frequency:", len(rules_freq_icon))
            print(json.dumps(rules_freq_icon, indent=3))
            print(json.dumps(edge_freq_icon, indent=3))

        return rules_freq_text, edge_freq_text, rules_freq_icon, edge_freq_icon

    def analyze_basic(self, results, verbose=False):
        def compute_accuracy(results):
            n_correct, n_correct_icons = 0, 0
            n_puzzles, n_puzzles_icon = 0, 0
            for result in results:
                graph_name = os.path.basename(result["image"]).split(".")[0]
                graph = self._graph_answer_pairs[graph_name]
                contains_icon = sum([1 if "icon" in attr else 0 for attr in get_node_attributes(graph).values()]) > 0
                if not contains_icon:
                    n_puzzles += 1
                    if result["is_correct"]:
                        n_correct += 1
                elif contains_icon:
                    n_puzzles_icon += 1
                    if result["is_correct"]:
                        n_correct_icons += 1

            return (n_correct / n_puzzles) * 100, (n_correct_icons / n_puzzles_icon) * 100

        def compute_max_answer_occurrence(results):
            answers_freq, answers_freq_icon = {}, {}
            for result in results:
                if "clean_output" not in result:
                    continue
                answer = list(result["clean_output"].keys())[0]
                graph_name = os.path.basename(result["image"]).split(".")[0]
                graph = self._graph_answer_pairs[graph_name]
                contains_icon = sum([1 if "icon" in attr else 0 for attr in get_node_attributes(graph).values()]) > 0
                if not contains_icon:
                    if answer not in answers_freq:
                        answers_freq[answer] = 0
                    answers_freq[answer] += 1
                elif contains_icon:
                    if answer not in answers_freq_icon:
                        answers_freq_icon[answer] = 0
                    answers_freq_icon[answer] += 1

            answers_freq = {answer: freq/sum(answers_freq.values()) for answer, freq in answers_freq.items()}
            answers_freq_icon = {answer: freq/sum(answers_freq_icon.values()) for answer, freq in answers_freq_icon.items()}
            if answers_freq != {} and answers_freq_icon != {}:
                max_answer_freq = {max(answers_freq, key=answers_freq.get): round(answers_freq[max(answers_freq, key=answers_freq.get)] * 100, 2)}
                max_answer_freq_icon = {max(answers_freq_icon, key=answers_freq_icon.get): round(answers_freq_icon[max(answers_freq_icon, key=answers_freq_icon.get)] * 100, 2)}
                return max_answer_freq, max_answer_freq_icon
            return None

        accuracy, accuracy_icons = compute_accuracy(results)
        most_common_answer, most_common_answer_icon = compute_max_answer_occurrence(results)

        if verbose:
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Most common answer: {most_common_answer}")
            print(f"Accuracy (icons): {accuracy_icons:.2f}")
            print(f"Most common answer (icon): {most_common_answer_icon}")

        return round(accuracy, 2), most_common_answer, round(accuracy_icons, 2), most_common_answer_icon


    def analyze_overall(self, basic_results, rule_results, verbose=False):
        def calculate_averages(freqs):
            sums_counts = {}
            for freq in freqs:
                for rule, result in freq.items():
                    if rule not in sums_counts:
                        sums_counts[rule] = [result[0], 1]
                    sums_counts[rule][0] += result[0]
                    sums_counts[rule][1] += 1

            averages = {rule: round(sums_counts[rule][0] / sums_counts[rule][1], 2) for rule in sums_counts}
            return averages

        table_all_prompts = {}
        for prompt in ["1", "2", "3", "4"]:
            results_no_icon = {model: result[0] for model, result in basic_results[prompt].items() if result is not None}
            results_icon = {model: result[2] for model, result in basic_results[prompt].items() if result is not None}
            table_all_prompts[f"no_icon_prompt_{prompt}"] = results_no_icon
            table_all_prompts[f"icon_prompt_{prompt}"] = results_icon

        table_rules_per_prompt = {}
        for prompt in ["1", "2", "3", "4"]:
            results = [result for result in list(rule_results[prompt].values()) if result is not None]
            rules_freq_text = calculate_averages(np.array(results)[:, 0].tolist())
            edge_freq_text = calculate_averages(np.array(results)[:, 1].tolist())
            rules_freq_icon = calculate_averages(np.array(results)[:, 2].tolist())
            edge_freq_icon = calculate_averages(np.array(results)[:, 3].tolist())
            rules_freq_icon = {rule: ("-" if str(rule).startswith("direction") else freq) for rule, freq in
                               rules_freq_icon.items()}
            table_rules_per_prompt[f"no_icon_prompt_{prompt}"] = rules_freq_text
            table_rules_per_prompt[f"no_icon_prompt_{prompt}"].update(edge_freq_text)
            table_rules_per_prompt[f"icon_prompt_{prompt}"] = rules_freq_icon
            table_rules_per_prompt[f"icon_prompt_{prompt}"].update(edge_freq_icon)

        table_prompt_2 = pd.DataFrame(basic_results["2"]).transpose().drop(["mistral-7b"])
        table_prompt_2 = table_prompt_2.rename(columns={0: "accuracy (%)", 1: "most common answer (%)",
                                                        2: "accuracy (%)", 3: "most common answer (%)"})
        multi_columns = pd.MultiIndex.from_tuples(
            [('Without icons', 'accuracy (%)'), ('Without icons', 'most common answer (%)'),
             ('With icons', 'accuracy (%)'), ('With icons', 'most common answer (%)')]
        )



        table_prompt_2.columns = multi_columns
        if verbose:
            print(table_prompt_2)

        table_rules_per_prompt = pd.DataFrame(table_rules_per_prompt)
        if verbose:
            print("\nAccuracy per puzzle including a specified rule (Individual + Relational + Modifier)")
            print(table_rules_per_prompt)

        table_all_prompts = pd.DataFrame.from_dict(table_all_prompts)
        if verbose:
            print("\nAccuracy per prompt")
            print(table_all_prompts)

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
            output = output.lower()
            if output in answer_to_letter:
                letter = answer_to_letter[output]
                is_correct = {letter: output} == result["correct"]
                result["is_correct"] = is_correct
                result["clean_output"] = {letter: output}
            else:
                result["is_correct"] = False
        return result

    def _preprocess_llava_13b_result(self, result):
        result["output"] = re.split("ASSISTANT: ", result["output"])[1]
        return result

    def _preprocess_llava_34b_result(self, result, counter):
        output = result["output"]
        if "<|im_start|> assistant\n" in output:
            output = output.split("<|im_start|> assistant\n")[1].strip()
            match = re.search(r"\(\((.*?)\)\)", output)
            if match:
                result["output"] = match.group(1)
                counter += 1
                return result, counter
            result["output"] = output
            counter += 1
            return result, counter
        return result, counter

    def _preprocess_fuyu_result(self, result, counter):
        output = result["output"]
        if "" in output:
            output = output.split("")[1].strip()
            match = re.search(r"\(\((.*?)\)\)", output)
            if match:
                result["output"] = match.group(1)
                counter += 1
                return result, counter
            result["output"] = output
            counter += 1
            return result, counter

        return result, counter

    def _preprocess_mistral_result(self, result, counter, is_phrase=False):
        output = result["output"]
        match = re.search(r"\(\((.*?)\)\)", output)
        if match:
            result["output"] = match.group(1)
            counter += 1
            return result, counter
        return result, counter

    def _preprocess_cogvlm_result(self, result, counter):
        output = result["output"]
        output = output.replace("</s>", "")
        result["output"] = output
        return result, counter

    def _preprocess_qwenvl_result(self, result, counter):
        output = result["output"]
        if len(re.findall(r'\(\((.*?)\)\)', output)) > 0:
            result["output"] = re.findall(r'\(\((.*?)\)\)', output)[0]
        elif len(re.findall(r"is \"(.*?)\".", output)) > 0:
            result["output"] = re.findall(r"is \"(.*?)\".", output)[0]
        elif len(re.findall(r"is \"(.*?).\"", output)) > 0:
            result["output"] = re.findall(r"is \"(.*?).\"", output)[0]

        return result, counter
