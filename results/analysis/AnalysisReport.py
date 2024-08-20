import glob
import json
import os
import re
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd

from puzzles.patterns.Rule import Rule
from util import get_node_attributes, get_answer_graph_pairs


class AnalysisReport:
    """
    Class to analyze the results from all the models.
    """

    def __init__(self):
        self.results_dir = f"{os.path.dirname(__file__)}/results"
        self._graph_answer_pairs = get_answer_graph_pairs(combine=True)
        self._model_types = {
            "non_instruction": ["blip-2_opt-2.7b", "blip-2_opt-6.7b", "fuyu-8b"],
            "instruction": ["instructblip", "llava-1.5-13b", "blip-2_flan-t5-xxl",
                            "cogvlm", "qwenvl", "mistral-7b", "llava-1.6-34b", "gpt-4o",
                            "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro"]
        }
        self._prompt_types = ["1", "2", "3", "4"]

    def prepare_results_data(self, model_types):
        """
        Prepares a dictionary with some of the results on model performance and rules. This is done by setting
        some results to blank ("-").
        :param model_types: list of models.
        :return: pair of dictionaries: one for model performance on each prompt, and the other on different rules.
        """

        all_basic_results = {prompt: {model: None} for model, prompt in product(*[model_types, self._prompt_types])}
        all_rule_results = {prompt: {model: None} for model, prompt in product(*[model_types, self._prompt_types])}

        # Mistral results
        all_basic_results["1"]["mistral-7b"] = ["-"] * 6
        all_basic_results["2"]["mistral-7b"] = ["-"] * 6

        # Human, BC, FC results
        for i in ["1", "3", "4"]:
            all_basic_results[i]["human"] = ["-"] * 6
            all_basic_results[i]["belief_graphs_gpt-4o"] = ["-"] * 6
            all_basic_results[i]["belief_graphs_gpt-4o-mini"] = ["-"] * 6
            all_basic_results[i]["gpt-4o-fc"] = ["-"] * 6
            all_basic_results[i]["gpt-4o-mini-fc"] = ["-"] * 6
            all_basic_results[i]["gemini-1.5-pro-fc"] = ["-"] * 6
            all_basic_results[i]["gemini-1.5-flash-fc"] = ["-"] * 6

        return all_basic_results, all_rule_results

    def generate_all(self):
        """
        Analyzes all the models.
        """
        model_types = self._model_types["non_instruction"] + self._model_types["instruction"] + ["clip"]
        all_basic_results, all_rule_results = self.prepare_results_data(model_types)

        # Results for basic models
        for model, prompt in product(*[model_types, self._prompt_types]):
            if model == "mistral-7b" and (prompt == "1" or prompt == "2"):
                continue
            basic_results, rule_results = self.generate(model, prompt)
            all_basic_results[prompt][model] = basic_results
            if (model != "blip-2_opt-2.7b" and model != "blip-2_opt-6.7b" and model != "instructblip"
                    and model != "mistral-7b"):
                all_rule_results[prompt][model] = rule_results

        # Results for belief graphs
        all_basic_results["2"]["belief_graphs_gpt-4o"] = self.generate("belief_graphs_gpt-4o",
                                                                       prompt_type="N/A")[0]
        all_basic_results["2"]["belief_graphs_gpt-4o-mini"] = self.generate("belief_graphs_gpt-4o-mini",
                                                                            prompt_type="N/A")[0]

        # Results for human performance
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

        table_prompt_2, table_all_prompts, table_rules_per_prompt, table_rules_gpt4o = (
            self.analyze_overall(all_basic_results, all_rule_results))

        # Print results
        print("\nMain table (accuracy per model for prompt 2). There are some slight differences due to randomness.")
        print(table_prompt_2)
        print("\nAccuracy per prompt for each model. There are some slight differences due to randomness.")
        print(table_all_prompts)
        print("\nPercentage of puzzles solved including a specified rule (Individual + Relational + Modifier)\n"
              "(averaged across all models)")
        print(table_rules_per_prompt)
        print("\nPercentage of puzzles solved including a specified rule (Individual + Relational + Modifier)\n"
              "(GPT-4o)")
        print(table_rules_gpt4o)

    def generate(self, model_type, prompt_type, mistral_type=None):
        """
        Loads the results for a model based on the specified model and prompt type and analyzes it.

        :param model_type: string denoting which model to load. Either:
        ["blip-2_opt-2.7b", "blip-2_opt-6.7b", "fuyu-8b", "instructblip", "llava-1.5-13b", "blip-2_flan-t5-xxl",
        "cogvlm", "qwenvl", "mistral-7b", "llava-1.6-34b", "gpt-4o", "gpt-4o-mini", "gemini-1.5-flash",
        "gemini-1.5-pro"]
        :param prompt_type: string denoting which prompt to laod. Either: ["1", "2", "3", "4"]
        :param mistral_type: string denoting which mistral type is used.
        :return: a tuple of results: first value is model performance on all prompts and second value is the model
        performance on each rule.
        """

        # Load model results
        if model_type == "clip":
            with open(f"{self.results_dir}/{model_type}.json", "r") as file:
                results = json.load(file)["results"]
        elif model_type in ["belief_graphs_gpt-4o-mini", "belief_graphs_gpt-4o",
                            "belief_graphs_gemini-1.5-flash", "belief_graphs_gemini-1.5-pro"]:
            with open(f"{self.results_dir}/belief_graphs/{model_type}.json", "r") as file:
                results = json.load(file)["results"]
        elif model_type in ["gpt-4o", "gpt-4o-mini", "gemini-1.5-flash", "gemini-1.5-pro"]:
            with open(f"{self.results_dir}/prompt_{prompt_type}/closed_source_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)[model_type]
        elif model_type == "mistral-7b":
            with open(f"{self.results_dir}/{model_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)["results"]
        else:
            with open(f"{self.results_dir}/prompt_{prompt_type}/{model_type}_prompt_{prompt_type}.json", "r") as file:
                results = json.load(file)["results"]

        # Preprocess results (standardization)
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

        # Analyze model performance on different prompts
        basic_results = self.analyze_basic(results)

        # Analyze model performance on different rules
        rule_results = self.analyze_by_rule(results)

        return basic_results, rule_results

    def analyze_by_rule(self, results):
        """
        Computes the accuracy on each rule for a specific model (and sample size of that rule) split over node
        (individual + modifier) or edge (relational) rules, for both text + icon puzzles separately.
        :param results: results of a model.
        :return: a tuple in the following format: (performance on node rules for text puzzles, performance on edge
        rules for text puzzles, performance on node rules for icon puzzles, performance onedge rules for icon puzzles)
        """

        rules = list(Rule.get_all_rules()["individual"].keys()) + ["sound"]
        rules_freq_text, rules_freq_icon = {}, {}
        edge_freq_text, edge_freq_icon = {}, {}

        def increment_rule_freq(rule, value, contains_icons):
            """
            Increments the number of puzzles solved for each rule (if it is correctly solved).
            :param rule: rule category (e.g., 'direction , 'highlight').
            :param value: rule (e.g., 'up', 'down').
            :param contains_icons: flag to denote if the puzzle contains an icon or not.
            """

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
            """
            Formats the attributes of a node.
            :param attrs: attributes of node.
            :return: reformatted attributes of node.
            """

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

        # Iterate over each result
        for result in results:
            graph_name = os.path.basename(result["image"]).split(".")[0]
            if graph_name not in self._graph_answer_pairs:
                continue

            # Get the graph corresponding to a puzzle
            graph = self._graph_answer_pairs[graph_name]
            node_attrs = get_node_attributes(graph)

            # Check if the puzzle contains an icon
            contains_icons = sum([1 if "icon" in attr else 0 for attr in node_attrs.values()]) > 0

            # Compute performance on this puzzle by the rules presents for the nodes (individual + modifier rules)
            for node, attrs in node_attrs.items():
                attrs_ = format_attrs(attrs)
                for rule, value in attrs_.items():
                    increment_rule_freq(rule, value, contains_icons)

            # Compute performance on this puzzle by the rules presents for the edges (relational rules)
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

        # Converts the number of puzzles correctly solved by rule to a percentage
        for rule, freq in rules_freq_text.items():
            rules_freq_text[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))
        for rule, freq in edge_freq_text.items():
            edge_freq_text[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))
        for rule, freq in rules_freq_icon.items():
            rules_freq_icon[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))
        for rule, freq in edge_freq_icon.items():
            edge_freq_icon[rule] = (round((sum(freq) / len(freq)) * 100, 2), len(freq))

        return rules_freq_text, edge_freq_text, rules_freq_icon, edge_freq_icon

    def analyze_basic(self, results):
        """
        Analyzes model performance on each prompt by computing the accuracy and answer distributions.

        :param results: results for a model.
        :return: tuple in the following format: (accuracy on text puzzles, most common answer on text puzzles,
        accuracy on icon puzzles, most common answer on icon puzzles, accuracy on text overlap puzzles,
        accuracy on icon overlap puzzles).
        """

        def compute_accuracy(results):
            """
            Computes the percentage of correctly solved puzzles for text/icon puzzles.
            :param results: results for a model
            :return: tuple of the form: (accuracy on text puzzles, accuracy on icon puzzles)
            """
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
            """
            Computes the accuracy on overlapping text + icon puzzles.

            :param results: results for a model.
            :return: tuple of the form: (accuracy on text overlap puzzles, accuracy on icon overlap puzzles).
            """
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
            """
            Computes the most common answer.

            :param results: results for a model.
            :return: tuple of the form: (most common answer on text puzzles, most common answer on icon puzzles).
            """
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

            # Convert frequencies to a percentage
            answers_freq = {answer: freq / sum(answers_freq.values()) for answer, freq in answers_freq.items()}
            answers_freq_icon = {answer: freq / sum(answers_freq_icon.values()) for answer, freq in
                                 answers_freq_icon.items()}

            # Get the highest percentage for text and icon puzzles
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

    def analyze_overall(self, basic_results, rule_results):
        """
        Analyzes all models and formats them into a dataframe.

        :param basic_results: model performance on each prompt.
        :param rule_results: model performance on each rule.
        :return: a tuple of Dataframes in the format: (prompt 2 results, all prompts results, rule results averaged
        across all models, rule results for GPT-4o).
        """
        def calculate_averages(freqs):
            """
            Helper function to average model performance on rules.

            :param freqs: rule results for models.
            :return: dictionary mapping each rule to an average percentage of puzzles solved.
            """
            sums_counts = {}
            for freq in freqs:
                for rule, result in freq.items():
                    if rule not in sums_counts:
                        sums_counts[rule] = [result[0], 1]
                    sums_counts[rule][0] += result[0]
                    sums_counts[rule][1] += 1

            averages = {rule: round(sums_counts[rule][0] / sums_counts[rule][1], 2) for rule in sums_counts}
            return averages

        basic_results = self._add_results_from_paper(basic_results)

        # Create dataframe for all prompts results
        table_all_prompts = {}
        for prompt in ["1", "2", "3", "4"]:
            results_no_icon = {model: result[0] for model, result in basic_results[prompt].items() if
                               result is not None if model != "clip"}
            results_icon = {model: result[2] for model, result in basic_results[prompt].items() if
                            result is not None if model != "clip"}
            table_all_prompts[f"no_icon_prompt_{prompt}"] = results_no_icon
            table_all_prompts[f"icon_prompt_{prompt}"] = results_icon

        # Create dataframe for rule results averaged across all models
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

        # Create dataframe for rule results for GPT-4o
        table_rules_per_prompt_gpt_4o = {}
        results = rule_results["2"]["gpt-4o"]
        rules_freq_text = {rule: value[0] for rule, value in results[0].items()}
        edge_freq_text = {rule: value[0] for rule, value in results[1].items()}
        rules_freq_icon = {rule: value[0] for rule, value in results[2].items()}
        edge_freq_icon = {rule: value[0] for rule, value in results[3].items()}
        rules_freq_icon = {rule: ("-" if str(rule).startswith("direction") else freq) for rule, freq in
                           rules_freq_icon.items()}
        table_rules_per_prompt_gpt_4o[f"no_icon_prompt_2"] = rules_freq_text
        table_rules_per_prompt_gpt_4o[f"no_icon_prompt_2"].update(edge_freq_text)
        table_rules_per_prompt_gpt_4o[f"icon_prompt_2"] = rules_freq_icon
        table_rules_per_prompt_gpt_4o[f"icon_prompt_2"].update(edge_freq_icon)

        # Create dataframe for prompt 2 rule results
        table_prompt_2 = pd.DataFrame(basic_results["2"]).transpose().drop(["mistral-7b"])
        table_prompt_2 = table_prompt_2.rename(columns={0: "accuracy (%)", 1: "most common answer (%)",
                                                        2: "accuracy (%)", 3: "most common answer (%)"})

        table_prompt_2 = table_prompt_2.drop([4, 5], axis=1)
        multi_columns = pd.MultiIndex.from_tuples(
            [('Without icons', 'accuracy (%)'), ('Without icons', 'most common answer (%)'),
             ('With icons', 'accuracy (%)'), ('With icons', 'most common answer (%)')]
        )

        # Conversion to dataframes
        table_prompt_2.columns = multi_columns
        table_all_prompts = pd.DataFrame.from_dict(table_all_prompts)
        table_rules_per_prompt = pd.DataFrame(table_rules_per_prompt)
        table_rules_per_prompt_gpt_4o = pd.DataFrame(table_rules_per_prompt_gpt_4o)
        return table_prompt_2, table_all_prompts, table_rules_per_prompt, table_rules_per_prompt_gpt_4o

    def analyze_non_icon_vs_icon(self):
        """
        Gets all overlapping puzzles that have both a text and icon variant.

        :return: tuple of the form: (overlapping text puzzles, overlapping icon puzzles)
        """
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

    def _add_results_from_paper(self, basic_results):
        basic_results["2"]["gpt-4o"] = list(basic_results["2"]["gpt-4o"])
        basic_results["2"]["gpt-4o"][1] = {"A": 26.8}
        basic_results["2"]["gpt-4o"][3] = {"B": 29.1}

        basic_results["2"]["gpt-4o-mini"] = list(basic_results["2"]["gpt-4o-mini"])
        basic_results["2"]["gpt-4o-mini"][1] = {"B": 29.4}
        basic_results["2"]["gpt-4o-mini"][3] = {"B": 31.3}

        basic_results["2"]["gemini-1.5-pro"] = list(basic_results["2"]["gemini-1.5-pro"])
        basic_results["2"]["gemini-1.5-pro"][1] = {"D": 31.3}
        basic_results["2"]["gemini-1.5-pro"][3] = {"D": 34.8}

        basic_results["2"]["gemini-1.5-flash"] = list(basic_results["2"]["gemini-1.5-flash"])
        basic_results["2"]["gemini-1.5-flash"][1] = {"C": 32.5}
        basic_results["2"]["gemini-1.5-flash"][3] = {"B": 34.4}

        basic_results["2"]["gpt-4o-fc"] = (82.9, {"D": 28.8}, 79.8, {"D": 30.1}, "-", "-")
        basic_results["2"]["gpt-4o-mini-fc"] = (74.3, {"C": 27.7}, 72.8, {"D": 27.4}, "-", "-")
        basic_results["2"]["gemini-1.5-pro-fc"] = (74.2, {"D": 34.6}, 76.3, {"D": 33.0}, "-", "-")
        basic_results["2"]["gemini-1.5-flash-fc"] = (70.0, {"D": 29.5}, 80.9, {"C": 30.1}, "-", "-")
        return basic_results

    def _standardize_general_result(self, result, mistral_type=None):
        """
        Standardizes the output of a result and checks if it is correct using regex matching.

        :param result: single result for a model.
        :param mistral_type: string denoting the mistral type.
        :return: result for a puzzle with the added cleaned output and a boolean flag to denote if that puzzle
        was correctly solved.
        """
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
        """
        Preprocess Llava 13b result.
        :param result: result for Llava 13b.
        :return: standardized result for Llava 13b.
        """
        result["output"] = re.split("ASSISTANT: ", result["output"])[1]
        return result

    def _preprocess_llava_34b_result(self, result):
        """
        Preprocess Llava 34b result.
        :param result: result for Llava 34b.
        :return: standardized result for Llava 34b.
        """
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
        """
        Preprocess Fuyu-8b result.
        :param result: result for Fuyu-8b.
        :return: standardized result for Fuyu-8b.
        """
        output = result["output"]
        if "" in output:
            output = output.split("")[1].strip()
            if output.startswith("A:"):
                output = output.split("A:", 1)[1].strip()
            result["output"] = output
            return result

        return result

    def _preprocess_mistral_result(self, result):
        """
        Preprocess Mistral result.
        :param result: result for Mistral.
        :return: standardized result for Mistral.
        """
        output = result["output"]
        match = re.search(r"\(\((.*?)\)\)", output)
        if match:
            result["output"] = match.group(1)
            return result
        return result

    def _preprocess_cogvlm_result(self, result):
        """
        Preprocess CogVLM result.
        :param result: result for CogVLM.
        :return: standardized result for CogVLM.
        """
        output = result["output"]
        output = output.replace("</s>", "")
        result["output"] = output
        return result

    def _preprocess_qwenvl_result(self, result):
        """
        Preprocess QwenVL result.
        :param result: result for QwenVL.
        :return: standardized result for QwenVL.
        """
        output = result["output"]
        if len(re.findall(r'\(\((.*?)\)\)', output)) > 0:
            result["output"] = re.findall(r'\(\((.*?)\)\)', output)[0]
        elif len(re.findall(r"is \"(.*?)\".", output)) > 0:
            result["output"] = re.findall(r"is \"(.*?)\".", output)[0]
        elif len(re.findall(r"is \"(.*?).\"", output)) > 0:
            result["output"] = re.findall(r"is \"(.*?).\"", output)[0]

        return result

    def _preprocess_closed_source_result(self, result):
        """
        Preprocess closed source model result (for GPT-4o (mini), and Gemini-1.5-(pro/flash)). The answer distributions
        are added later.
        :param result: result for closed source model.
        :return: standardized result for closed source model.
        """
        result["is_correct"] = True if result["label"] == "correct" else False
        result["clean_output"] = {"A": 0.0}
        return result
