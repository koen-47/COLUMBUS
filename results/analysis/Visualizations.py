import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


class Visualizations:
    def __init__(self):
        pass

    def visualize_rule_frequency_gpt4o(self, gpt_4o_rule_results):
        rules = {
            "highlight_before": [],
            "highlight_middle": [],
            "highlight_after": [],
            "direction_up": [],
            "direction_down": [],
            "direction_reverse": [],
            "size_big": [],
            "size_small": [],
            "color": [],
            "cross": [],
            "next-to": [],
            "inside": [],
            "above": [],
            "outside": [],
            "sound": [],
            "repetition_two": [],
            "repetition_four": []
        }

        for i in range(4):
            for rule, values in gpt_4o_rule_results[i].items():
                if i >= 2 and rule.startswith("direction"):
                    rules[rule.lower()].append("-")
                else:
                    rules[rule.lower()].append(values[0])

        print(json.dumps(rules, indent=3))

        def plot_rule_frequency(ax, data):
            for i, (acc_1, acc_2) in enumerate(data.values()):
                if acc_2 == "-":
                    ax.bar(i + 1, acc_1, 0.35, color="#0077b3", label="COLUMBUS-text" if i == 0 else "")
                else:
                    ax.bar((i + 1) - 0.2, acc_1, 0.35, color="tab:blue", label="COLUMBUS-text" if i == 0 else "")
                    ax.bar((i + 1) + 0.2, acc_2, 0.35, color="tab:orange", label="COLUMBUS-icon" if i == 0 else "")

            ax.set_xticks(np.arange(1, len(data) + 1))
            ax.set_xticklabels(["" for _ in range(len(data))])
            ax.set_xlim(0.5, len(data) + 0.5)
            ax.set_ylim(0, 100)
            ax.set_axisbelow(True)
            ax.spines[['right', 'top']].set_visible(False)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 16

        fig = plt.figure(figsize=(12.5, 6))
        fig.subplots_adjust(left=0.075, right=0.981, top=0.827, bottom=0.581)
        ax = fig.add_subplot(1, 1, 1)
        plot_rule_frequency(ax, rules)
        plt.ylabel("Accuracy (%)", fontweight="bold")
        plt.legend(frameon=False, bbox_to_anchor=(0.25, 1.0), ncol=2)
        plt.show()
        plt.close()

    def visualize_rule_frequency_average(self, table_rules_per_prompt):
        rules = {
            "individual": {
                "highlight_before": [],
                "highlight_middle": [],
                "highlight_after": [],
                "direction_up": [],
                "direction_down": [],
                "direction_reverse": [],
                "size_big": [],
                "size_small": [],
                "color": [],
                "cross": []
            },
            "relational": {
                "next-to": [],
                "inside": [],
                "above": [],
                "outside": []
            },
            "modifier": {
                "sound": [],
                "repetition_two": [],
                "repetition_four": []
            }
        }

        prompt_2_rule_results = table_rules_per_prompt[["no_icon_prompt_2", "icon_prompt_2"]]
        for rule, values in prompt_2_rule_results.iterrows():
            values = tuple(values.tolist())
            if rule in rules["individual"].keys():
                rules["individual"][rule] = values
            elif rule in rules["modifier"].keys():
                rules["modifier"][rule] = values
            else:
                rules["relational"][rule.lower()] = values

        print(json.dumps(rules, indent=3))

        def plot_rule_frequency(ax, data):
            data = list(data["individual"].values()) + list(data["relational"].values()) + list(
                data["modifier"].values())

            for i, (acc_1, acc_2) in enumerate(data):
                if acc_2 == "-":
                    ax.bar(i + 1, acc_1, 0.35, color="#0077b3", label="Puzzles without icons" if i == 0 else "")
                else:
                    ax.bar((i + 1) - 0.2, acc_1, 0.35, label="Puzzles without icons" if i == 0 else "")
                    ax.bar((i + 1) + 0.2, acc_2, 0.35, label="Puzzles with icons" if i == 0 else "")

            ax.set_xticks(np.arange(1, len(data) + 1))
            ax.set_xticklabels(["" for _ in range(len(data))])
            ax.set_xlim(0.5, len(data) + 0.5)
            ax.set_ylim(0, 100)
            ax.set_axisbelow(True)
            ax.spines[['right', 'top']].set_visible(False)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 16

        fig = plt.figure(figsize=(12.5, 6))
        ax = fig.add_subplot(1, 1, 1)
        plot_rule_frequency(ax, rules)
        plt.ylabel("Accuracy (%)", fontweight="bold")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.close()

    def visualize_prompts(self, table_all_prompts):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 27

        models = ["gpt-4o", "fuyu-8b", "blip-2_flan-t5-xxl", "mistral-7b"]
        model_names = dict(zip(models, ["GPT-4o", "Fuyu-8b", "BLIP-2 Flan-T5-XXL", "Mistral-7b"]))
        data = table_all_prompts.loc[models]
        non_icon_prompt_data_per_model = {model: [] for model in models}
        icon_prompt_data_per_model = {model: [] for model in models}
        for prompt in ["1", "2", "3", "4"]:
            prompt_data = np.array(data[[col for col in data.columns if col.endswith(prompt)]].reset_index(drop=True))
            prompt_data = prompt_data[:-1] if prompt_data.tolist()[-1] == ["-", "-"] else prompt_data
            non_icon_prompt_data = dict(zip(models[:len(prompt_data)], np.array(prompt_data)[:, 0]))
            icon_prompt_data = dict(zip(models[:len(prompt_data)], np.array(prompt_data)[:, 1]))
            for model, acc in non_icon_prompt_data.items():
                non_icon_prompt_data_per_model[model].append(acc)
            for model, acc in icon_prompt_data.items():
                icon_prompt_data_per_model[model].append(acc)

        def plot(ax, prompt_data, is_icon=False):
            markers = ['o', 's', '^', 'D']
            lines = []
            for i, (model, accuracy) in enumerate(prompt_data.items()):
                if model == "mistral-7b":
                    line, = ax.plot(list(range(3, 5)), accuracy, label=model_names[model], marker=markers[i], linewidth=3,
                            markersize=12)
                else:
                    line, = ax.plot(list(range(1, 5)), accuracy, label=model_names[model], marker=markers[i], linewidth=3,
                            markersize=12)
                lines.append(line)

            if not is_icon:
                line = ax.axhline(y=99.0, linestyle='--', linewidth=3, label="Human", color="magenta")
            else:
                line = ax.axhline(y=91.5, linestyle='--', linewidth=3, label="Human", color="magenta")
            lines.append(line)

            ax.grid()
            ax.set_ylim(0, 100)
            ax.set_xlabel("Prompt #")
            if not is_icon:
                ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(list(range(1, 5)))
            return lines

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.subplots_adjust(left=0.240, right=0.9, top=0.9, bottom=0.565)  # Adjust as needed

        plot(ax1, non_icon_prompt_data_per_model, is_icon=False)
        lines = plot(ax2, icon_prompt_data_per_model, is_icon=True)
        labels = [line.get_label() for line in lines]

        fig.legend(lines, labels, loc='lower center', shadow=True, ncol=5, fontsize=24)
        plt.show()
