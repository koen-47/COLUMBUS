import glob
import json
import os
import re
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import wordfreq
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import seaborn as sns

from puzzles.legacy.RebusGraphParser import RebusGraphParser
from puzzles.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from puzzles.patterns.Rule import Rule
from util import get_node_attributes, get_answer_graph_pairs
from results.analysis.Visualizations import Visualizations


class AnalysisReport:
    def __init__(self):
        self.results_dir = f"{os.path.dirname(__file__)}/results_v3"
        self._graph_answer_pairs = get_answer_graph_pairs("v3", combine=True)
        self._model_types = {"non_instruction": ["blip-2_opt-2.7b", "blip-2_opt-6.7b", "fuyu-8b"],
                             "instruction": ["instructblip", "llava-1.5-13b", "blip-2_flan-t5-xxl",
                                             "cogvlm", "qwenvl", "mistral-7b", "llava-1.6-34b", "gpt-4o",
                                             "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro"]}
        # "llava-1.6-34b"
        self._implemented_models = ["belief_graphs_gpt-4o-mini"]
        self._prompt_types = ["1", "2", "3", "4"]

    def generate_all(self, verbose=False):
        all_model_types = self._model_types["non_instruction"] + self._model_types["instruction"] + ["clip"]
        all_basic_results = {prompt: {model: None} for model, prompt in product(*[all_model_types, self._prompt_types])}
        all_rule_results = {prompt: {model: None} for model, prompt in product(*[all_model_types, self._prompt_types])}

        all_basic_results["1"]["mistral-7b"] = ["-"] * 6
        all_basic_results["2"]["mistral-7b"] = ["-"] * 6
        for i in ["1", "3", "4"]:
            all_basic_results["4"]["human"] = ["-"] * 6
            all_basic_results[i]["gpt-4o"] = ["-"] * 6
            all_basic_results[i]["gpt-4o-mini"] = ["-"] * 6
            all_basic_results[i]["gemini-1.5-flash"] = ["-"] * 6
            all_basic_results[i]["gemini-1.5-pro"] = ["-"] * 6

        for model, prompt in product(*[all_model_types, self._prompt_types]):
            if model == "mistral-7b" and (prompt == "1" or prompt == "2"):
                continue
            if model in ["gpt-4o", "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro"] and (prompt in ["1", "3", "4"]):
                continue
            basic_results, rule_results = self.generate(model, prompt, verbose=verbose)
            all_basic_results[prompt][model] = basic_results
            if (model != "blip-2_opt-2.7b" and model != "blip-2_opt-6.7b" and model != "instructblip"
                    and model != "mistral-7b"):
                all_rule_results[prompt][model] = rule_results

        # print(self.generate("clip", prompt_type="N/A"))
        print(self.generate("belief_graphs_gpt-4o", prompt_type="N/A"))
        print(self.generate("belief_graphs_gpt-4o-mini", prompt_type="N/A"))
        # print(self.generate("belief_graphs_gemini-1.5-flash", prompt_type="N/A"))
        # print(self.generate("belief_graphs_gemini-1.5-pro", prompt_type="N/A"))

        human_results = []
        for file_path in glob.glob(f"{self.results_dir}/human/*"):
            with open(file_path, "r") as file:
                results = json.load(file)
            for result in results:
                result = self._standardize_general_result(result)
            human_results.append(self.analyze_basic(results))
        human_results = ((human_results[0][0] + human_results[1][0]) / 2, human_results[0][1],
                         (human_results[0][2] + human_results[1][2]) / 2, human_results[0][3], "-", "-")
        all_basic_results["2"]["human"] = human_results

        table_prompt_2, table_all_prompts, table_rules_per_prompt = self.analyze_overall(all_basic_results,
                                                                                         all_rule_results, verbose=True)

        if verbose:
            print("\nMain table (accuracy per model for prompt 2)")
            print(table_prompt_2)
            print("\nAccuracy per prompt for each model")
            print(table_all_prompts)
            print("\nPercentage of puzzles solved including a specified rule (Individual + Relational + Modifier)")
            print(table_rules_per_prompt)

        self.visualize(table_prompt_2, table_all_prompts, table_rules_per_prompt, all_rule_results)

    def generate(self, model_type, prompt_type, mistral_type=None, verbose=False):
        if model_type == "clip":
            with open(f"{self.results_dir}/{model_type}.json", "r") as file:
                results = json.load(file)["results"]
        elif model_type in ["belief_graphs_gpt-4o-mini", "belief_graphs_gpt-4o",
                            "belief_graphs_gemini-1.5-flash", "belief_graphs_gemini-1.5-pro"]:
            with open(f"{self.results_dir}/belief_graphs/{model_type}.json", "r") as file:
                results = json.load(file)["results"]
        elif model_type in ["gpt-4o", "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro"]:
            with open(f"{self.results_dir}/prompt_2/closed_source_prompt_2.json", "r") as file:
                results = json.load(file)[model_type]
        elif model_type == "gpt-4o-mini" and prompt_type == 2:
            with open(f"{self.results_dir}/prompt_2/closed_source_prompt_2.json", "r") as file:
                results = json.load(file)["results"]["gpt-4o-mini"]
        elif model_type == "mistral-7b":
            with open(f"{self.results_dir}/{model_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)["results"]
        else:
            with open(f"{self.results_dir}/prompt_{prompt_type}/{model_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)["results"]

        for result in results:
            if model_type in ["gpt-4o", "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro"]:
                result = self._preprocess_closed_source_result(result)
            else:
                if model_type == "llava-1.5-13b":
                    result = self._preprocess_llava_13b_result(result)
                elif model_type == "llava-1.6-34b":
                    result = self._preprocess_llava_34b_result(result)
                elif model_type == "fuyu-8b":
                    result = self._preprocess_fuyu_result(result)
                elif model_type == "cogvlm":
                    result = self._preprocess_cogvlm_result(result)
                elif model_type == "qwenvl":
                    result = self._preprocess_qwenvl_result(result)
                elif model_type == "mistral-7b":
                    result = self._preprocess_mistral_result(result)
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
            if graph_name not in self._graph_answer_pairs:
                continue
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

        return rules_freq_text, edge_freq_text, rules_freq_icon, edge_freq_icon

    def analyze_basic(self, results, verbose=False):
        def compute_accuracy(results):
            n_correct, n_correct_icons = 0, 0
            n_puzzles, n_puzzles_icon = 0, 0
            for result in results:
                graph_name = os.path.basename(result["image"]).split(".")[0]
                if graph_name not in self._graph_answer_pairs:
                    continue
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

        def compute_accuracy_overlap(results):
            non_icon_overlap, icon_overlap = self.analyze_non_icon_vs_icon()
            n_non_icon_correct = 0
            n_icon_correct = 0
            for result in results:
                puzzle = os.path.basename(result["image"]).split(".")[0]
                if puzzle in non_icon_overlap and result["is_correct"]:
                    n_non_icon_correct += 1
                elif puzzle in icon_overlap and result["is_correct"]:
                    n_icon_correct += 1

            return (n_non_icon_correct / len(non_icon_overlap)) * 100, (n_icon_correct / len(icon_overlap)) * 100

        def compute_max_answer_occurrence(results):
            answers_freq, answers_freq_icon = {}, {}
            for result in results:
                if "clean_output" not in result:
                    continue
                answer = list(result["clean_output"].keys())[0]
                graph_name = os.path.basename(result["image"]).split(".")[0]
                if graph_name not in self._graph_answer_pairs:
                    continue
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

            answers_freq = {answer: freq / sum(answers_freq.values()) for answer, freq in answers_freq.items()}
            answers_freq_icon = {answer: freq / sum(answers_freq_icon.values()) for answer, freq in
                                 answers_freq_icon.items()}
            if answers_freq != {} and answers_freq_icon != {}:
                max_answer_freq = {max(answers_freq, key=answers_freq.get): round(
                    answers_freq[max(answers_freq, key=answers_freq.get)] * 100, 2)}
                max_answer_freq_icon = {max(answers_freq_icon, key=answers_freq_icon.get): round(
                    answers_freq_icon[max(answers_freq_icon, key=answers_freq_icon.get)] * 100, 2)}
                return max_answer_freq, max_answer_freq_icon
            return None

        accuracy, accuracy_icons = compute_accuracy(results)
        accuracy_overlap, accuracy_icons_overlap = compute_accuracy_overlap(results)
        most_common_answer, most_common_answer_icon = compute_max_answer_occurrence(results)

        return (round(accuracy, 2), most_common_answer, round(accuracy_icons, 2), most_common_answer_icon,
                round(accuracy_overlap, 2), round(accuracy_icons_overlap, 2))

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

        basic_results["1"]["gpt-4o"] = [76.4, None, 77.6, None]
        basic_results["3"]["gpt-4o"] = [90.7, None, 94.4, None]
        basic_results["4"]["gpt-4o"] = [91.0, None, 93.5, None]

        table_all_prompts = {}
        for prompt in ["1", "2", "3", "4"]:
            results_no_icon = {model: result[0] for model, result in basic_results[prompt].items() if
                               result is not None if model != "clip"}
            results_icon = {model: result[2] for model, result in basic_results[prompt].items() if
                            result is not None if model != "clip"}
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

        overlap_acc = np.array(table_prompt_2[[4, 5]].reset_index(drop=True))[:-1]
        print("Overlapping puzzle accuracy:", overlap_acc[:, 0].mean() - overlap_acc[:, 1].mean())

        table_prompt_2 = table_prompt_2.drop([4, 5], axis=1)
        multi_columns = pd.MultiIndex.from_tuples(
            [('Without icons', 'accuracy (%)'), ('Without icons', 'most common answer (%)'),
             ('With icons', 'accuracy (%)'), ('With icons', 'most common answer (%)')]
        )

        table_prompt_2.columns = multi_columns
        table_all_prompts = pd.DataFrame.from_dict(table_all_prompts)
        table_rules_per_prompt = pd.DataFrame(table_rules_per_prompt)
        return table_prompt_2, table_all_prompts, table_rules_per_prompt

    def analyze_non_icon_vs_icon(self):
        non_icon_overlap_puzzles = []
        icon_overlap_puzzles = []
        for answer, graph in reversed(self._graph_answer_pairs.items()):
            if answer.endswith("icon") or answer.endswith("non-icon"):
                non_icon_puzzle = "_".join(answer.split("_")[:-1]) + "_non-icon"
                icon_puzzle = "_".join(answer.split("_")[:-1]) + "_icon"
                if non_icon_puzzle in self._graph_answer_pairs.keys() and icon_puzzle in self._graph_answer_pairs.keys():
                    non_icon_overlap_puzzles.append(non_icon_puzzle)
                    icon_overlap_puzzles.append(icon_puzzle)

        return list(set(non_icon_overlap_puzzles)), list(set(icon_overlap_puzzles))

    def visualize(self, table_prompt_2, table_all_prompts, table_rules_per_prompt, all_rule_results):
        visualization = Visualizations()
        # visualization.visualize_rule_frequency(table_rules_per_prompt)
        # visualization.visualize_prompts(table_all_prompts)
        visualization.visualize_rule_frequency_gpt4o(all_rule_results["2"]["gpt-4o"])
        # visualization.visualize_rule_frequency_gpt4o(all_rule_results["2"]["gemini-1.5-flash"])
        # visualization.visualize_rule_frequency_gpt4o(all_rule_results["2"]["gemini-1.5-pro"])


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

    def _preprocess_llava_34b_result(self, result):
        output = result["output"]
        if "<|im_start|> assistant\n" in output:
            output = output.split("<|im_start|> assistant\n")[1].strip()
            match = re.search(r"\(\((.*?)\)\)", output)
            if match:
                result["output"] = match.group(1)
                return result
            result["output"] = output
            return result
        return result

    def _preprocess_fuyu_result(self, result):
        output = result["output"]
        if "" in output:
            output = output.split("")[1].strip()
            if output.startswith("A:"):
                output = output.split("A:", 1)[1].strip()
            result["output"] = output
            return result

        return result

    def _preprocess_mistral_result(self, result):
        output = result["output"]
        match = re.search(r"\(\((.*?)\)\)", output)
        if match:
            result["output"] = match.group(1)
            return result
        return result

    def _preprocess_cogvlm_result(self, result):
        output = result["output"]
        output = output.replace("</s>", "")
        result["output"] = output
        return result

    def _preprocess_qwenvl_result(self, result):
        output = result["output"]
        if len(re.findall(r'\(\((.*?)\)\)', output)) > 0:
            result["output"] = re.findall(r'\(\((.*?)\)\)', output)[0]
        elif len(re.findall(r"is \"(.*?)\".", output)) > 0:
            result["output"] = re.findall(r"is \"(.*?)\".", output)[0]
        elif len(re.findall(r"is \"(.*?).\"", output)) > 0:
            result["output"] = re.findall(r"is \"(.*?).\"", output)[0]

        return result

    def _preprocess_closed_source_result(self, result):
        result["is_correct"] = True if result["label"] == "correct" else False
        result["clean_output"] = {"A": "asdf"}
        return result
