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


class AnalysisReport:
    def __init__(self):
        self.results_dir = f"{os.path.dirname(__file__)}/results_v2"
        self._graph_answer_pairs = get_answer_graph_pairs(combine=True)
        self._model_types = {"non_instruction": ["blip-2_opt-2.7b", "blip-2_opt-6.7b", "fuyu-8b"],
                             "instruction": ["instructblip", "llava-1.5-13b", "blip-2_flan-t5-xxl", "llava-1.6-34b",
                                             "cogvlm", "qwenvl", "mistral-7b"]}
        self._prompt_types = ["1", "2", "3", "4"]
        self._names = {"clip": "CLIP",
                       "blip-2_opt-2.7b": "BLIP-2\nOPT-2.7b",
                       "blip-2_opt-6.7b": "BLIP-2\nOPT-6.7b",
                       "blip-2_flan-t5-xxl": "BLIP-2\nFlan-T5-XXL",
                       "fuyu-8b": "Fuyu-8b",
                       "instructblip": "InstructBLIP",
                       "llava-1.5-13b": "Llava-13b",
                       "llava-1.6-34b": "Llava-34b",
                       "cogvlm": "CogVLM",
                       "qwenvl": "Qwen-VL",
                       "mistral-7b": "Mistral-7b",
                       "baseline": "Baseline",
                       "non_instruction": "Non-instruction tuned models",
                       "instruction": "Instruction tuned models"}

    def generate_all(self, verbose=False):
        all_model_types = self._model_types["non_instruction"] + self._model_types["instruction"] + ["clip"]
        all_basic_results = {prompt: {model: None} for model, prompt in product(*[all_model_types, self._prompt_types])}
        all_rule_results = {prompt: {model: None} for model, prompt in product(*[all_model_types, self._prompt_types])}

        all_basic_results["1"]["mistral-7b"] = ["-"] * 4
        all_basic_results["2"]["mistral-7b"] = ["-"] * 4
        all_basic_results["1"]["human"] = ["-"] * 4
        all_basic_results["3"]["human"] = ["-"] * 4
        all_basic_results["4"]["human"] = ["-"] * 4

        self.generate("clip", prompt_type="N/A")
        for model, prompt in product(*[all_model_types, self._prompt_types]):
            if model == "mistral-7b" and (prompt == "1" or prompt == "2"):
                continue
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

        table_prompt_2, table_all_prompts, table_rules_per_prompt = self.analyze_overall(all_basic_results,
                                                                                         all_rule_results, verbose=True)

        if verbose:
            print("\nMain table (accuracy per model for prompt 2)")
            print(table_prompt_2)
            print("\nAccuracy per prompt for each model")
            print(table_all_prompts)
            print("\nPercentage of puzzles solved including a specified rule (Individual + Relational + Modifier)")
            print(table_rules_per_prompt)

        self._visualize(table_prompt_2, table_all_prompts, table_rules_per_prompt)

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
        most_common_answer, most_common_answer_icon = compute_max_answer_occurrence(results)

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

        basic_results["3"]["llava-1.6-34b"] = [60.00, None, 80.25, None]
        basic_results["4"]["llava-1.6-34b"] = [60.33, None, 85.28, None]
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
        multi_columns = pd.MultiIndex.from_tuples(
            [('Without icons', 'accuracy (%)'), ('Without icons', 'most common answer (%)'),
             ('With icons', 'accuracy (%)'), ('With icons', 'most common answer (%)')]
        )

        table_prompt_2.columns = multi_columns
        table_all_prompts = pd.DataFrame.from_dict(table_all_prompts)
        table_rules_per_prompt = pd.DataFrame(table_rules_per_prompt)
        return table_prompt_2, table_all_prompts, table_rules_per_prompt

    def _visualize(self, table_prompt_2, table_all_prompts, table_rules_per_prompt):
        mpl.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 13
        })

        human_prompt_2_data = {
            "no_icon": table_prompt_2["Without icons"]["accuracy (%)"]["human"],
            "icon": table_prompt_2["With icons"]["accuracy (%)"]["human"]
        }

        model_prompt_2_data = {
            "baseline": {"clip": (table_prompt_2["Without icons"]["accuracy (%)"]["clip"],
                                  table_prompt_2["With icons"]["accuracy (%)"]["clip"])},
            "non_instruction": {model: (acc_1, acc_2) for model, acc_1, acc_2 in
                                zip(table_prompt_2.index.values, table_prompt_2["Without icons"]["accuracy (%)"].values,
                                    table_prompt_2["With icons"]["accuracy (%)"].values)
                                if model in self._model_types["non_instruction"]},
            "instruction": {model: (acc_1, acc_2) for model, acc_1, acc_2 in
                            zip(table_prompt_2.index.values, table_prompt_2["Without icons"]["accuracy (%)"].values,
                                table_prompt_2["With icons"]["accuracy (%)"].values)
                            if model in self._model_types["instruction"]}
        }

        def mk_groups(data):
            try:
                newdata = data.items()
            except:
                return

            thisgroup = []
            groups = []
            for key, value in newdata:
                newgroups = mk_groups(value)
                if newgroups is None:
                    thisgroup.append((key, value))
                else:
                    thisgroup.append((key, len(newgroups[-1])))
                    if groups:
                        groups = [g + n for n, g in zip(newgroups, groups)]
                    else:
                        groups = newgroups
            return [thisgroup] + groups

        def add_vertical_line(ax, xpos, ypos):
            line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                              transform=ax.transAxes, color='black', linewidth=1.)
            line.set_clip_on(False)
            ax.add_line(line)

        def add_horizontal_line(ax, xpos, ypos):
            line = plt.Line2D([xpos + .1, xpos], [ypos, ypos],
                              transform=ax.transAxes, color='black', linewidth=2.)
            line.set_clip_on(False)
            ax.add_line(line)

        def label_group_bar(ax, data):
            groups = mk_groups(data)
            xy = groups.pop()
            x, y = zip(*xy)
            x = [self._names[label] for label in x]
            ly = len(y)
            xticks = np.arange(1, ly + 1)
            width = 0.375

            for i, (acc_1, acc_2) in enumerate(y):
                ax.bar((i + 1) - 0.2, acc_1, width, color="#0077b3", label="Puzzles without icons" if i == 0 else "")
                ax.bar((i + 1) + 0.2, acc_2, width, color="#008053", label="Puzzles with icons" if i == 0 else "")

            ax.axhline(human_prompt_2_data["no_icon"], linestyle="--", color="#0077b3", label="Human acc. (no icons)")
            ax.axhline(human_prompt_2_data["icon"], linestyle="--", color="#008053", label="Human acc. (icons)")

            ax.set_xticks(xticks)
            ax.set_xticklabels(x)
            ax.set_xlim(0.5, ly + 0.5)
            ax.set_ylim(0, 100)
            ax.set_axisbelow(True)
            ax.spines[['right', 'top']].set_visible(False)

            scale = 1. / ly
            for pos in range(ly + 1):
                add_vertical_line(ax, pos * scale, -.1)
            ypos = -.2
            while groups:
                group = groups.pop()
                pos = 0
                for label, rpos in group:
                    lxpos = (pos + .5 * rpos) * scale
                    ax.text(lxpos, ypos, self._names[label], ha='center', transform=ax.transAxes, fontweight="bold")
                    add_vertical_line(ax, pos * scale, ypos)
                    pos += rpos
                add_vertical_line(ax, pos * scale, ypos)
                ypos -= .1

        fig = plt.figure(figsize=(12.5, 6))
        ax = fig.add_subplot(1, 1, 1)
        label_group_bar(ax, model_prompt_2_data)
        plt.ylabel("Accuracy (%)", fontweight="bold")
        plt.legend(frameon=False, loc=(0.01, 0.65))
        plt.tight_layout()
        plt.savefig(f"{os.path.dirname(__file__)}/../../visualizations/models_prompt_2_results.png")
        plt.close()

        table_all_prompts_no_icon = table_all_prompts[[col for col in table_all_prompts if col.startswith("no_icon")]].drop(["mistral-7b", "human"])
        table_all_prompts_icon = table_all_prompts[[col for col in table_all_prompts if col.startswith("icon")]].drop(["mistral-7b", "human"])

        def generate_all_prompts_heatmap(ax, data):
            cmap = mcolors.LinearSegmentedColormap.from_list("n", ["#cc4100", "#EAEAF2", "#008053"])
            norm = plt.Normalize(0, 100)
            xtick_labels = [self._names[model] for model in table_all_prompts_no_icon.index]
            heatmap = sns.heatmap(data.to_numpy(dtype=float), cmap=cmap, norm=norm, linewidths=.5, ax=ax,
                                  yticklabels=xtick_labels, square=True, xticklabels=list(range(1, 5)))
            colorbar = heatmap.collections[0].colorbar
            colorbar.set_label("Accuracy (%)", weight="bold")
            heatmap.set_xlabel("Prompt", fontweight="bold")
            heatmap.axhline(3, color='white', lw=6.5)
            heatmap.text(-2.25, 1.5, "Non-instruction\ntuned models", rotation=90, ha="center", va="center", fontweight="bold")
            heatmap.text(-2.25, 6, "Instruction tuned models", rotation=90, ha="center", va="center", fontweight="bold")
            return heatmap

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6))
        generate_all_prompts_heatmap(ax1, table_all_prompts_no_icon)
        generate_all_prompts_heatmap(ax2, table_all_prompts_icon)
        plt.tight_layout()
        plt.savefig(f"{os.path.dirname(__file__)}/../../visualizations/models_all_prompts_results.png")
        plt.close()


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
