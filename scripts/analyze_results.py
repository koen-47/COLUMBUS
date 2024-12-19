import itertools
import json
import glob
import os.path
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

from extract_model_results import make_safe_prompt


def analyze_new_results():
    """
    Computes averages of the results.
    """

    prompt_2 = np.array([56.3, 21.8, 24.1, 32.0, 51.3, 57.8, 68.2, 58.0, 59.3, 52.0, 81.4, 74.8, 66.5, 61.6])
    prompt_2_icon = np.array([24.3, 25.7, 22.2, 51.4, 64.3, 72.2, 58.9, 60.3, 49.5, 84.9, 77.4, 70.4, 63.4])
    human, human_icon = 99.04, 91.51

    print("Analysis of overall performance (Table 2)")
    print(f"Avg. prompt 2:", prompt_2.mean())
    print(f"Avg. prompt 2 (icon):", prompt_2_icon.mean())
    print(f"Diff. human vs. prompt 2:", np.abs(prompt_2.mean() - human))
    print(f"Diff. human vs. prompt 2 (icon):", np.abs(prompt_2_icon.mean() - human_icon))

    prompt_1_four_models = np.array([32.6, 64.9, 76.4])
    prompt_1_four_models_icon = np.array([32.4, 72.2, 84.9])

    prompt_2_four_models = np.array([31.97, 68.2, 81.4])
    prompt_2_four_models_icon = np.array([31.08, 72.2, 84.9])

    prompt_3_four_models = np.array([43.0, 85.1, 90.7, 72.3])
    prompt_3_four_models_icon = np.array([47.3, 91.6, 94.4, 76.8])

    print("\nAnalysis of the four models from each category (Figure 5)")
    print(f"(four models) Diff. prompt 1 vs. prompt 2 (no icon):",
          prompt_2_four_models.mean() - prompt_1_four_models.mean())
    print(f"(four models) Diff. prompt 1 vs. prompt 2 (icon):",
          prompt_2_four_models_icon.mean() - prompt_1_four_models_icon.mean())

    print(f"(four models) Diff. prompt 2 vs. prompt 3 (no icon):",
          prompt_3_four_models.mean() - prompt_2_four_models.mean())
    print(f"(four models) Diff. prompt 2 vs. prompt 3 (icon):",
          prompt_3_four_models_icon.mean() - prompt_2_four_models_icon.mean())

    gpt4o_individual_prompt_2 = np.array([43.48, 78.95, 57.69, 72.73, 80.77, 70.0, 72.22, 86.96, 85.19, 66.67])
    gpt4o_relational_prompt_2 = np.array([84.64, 89.34, 90.72, 93.1])
    gpt4o_modifier_prompt_2 = np.array([78.2, 75.92, 69.35])

    gpt4o_individual_prompt_2_icon = np.array([30.0, 100.0, 80.0, 54.55, 70.0, 100.0, 83.33])
    gpt4o_relational_prompt_2_icon = np.array([82.3, 81.73, 86.84, 88.89])
    gpt4o_modifier_prompt_2_icon = np.array([81.33, 83.12, 82.14])

    print("\nAnalysis of GPT-4o on different rules (Figure 6)")
    print("(GPT-4o) Diff. between individual vs. relational accuracy (no icon):", gpt4o_individual_prompt_2.mean() -
          gpt4o_relational_prompt_2.mean())
    print("(GPT-4o) Diff. between individual vs. modifier accuracy (no icon):", gpt4o_individual_prompt_2.mean() -
          gpt4o_modifier_prompt_2.mean())

    print("(GPT-4o) Diff. between individual vs. relational accuracy (icon):", gpt4o_individual_prompt_2_icon.mean() -
          gpt4o_relational_prompt_2_icon.mean())
    print("(GPT-4o) Diff. between individual vs. modifier accuracy (no icon):", gpt4o_individual_prompt_2_icon.mean() -
          gpt4o_modifier_prompt_2_icon.mean())

    prompt_2_all_models = np.array([56.3, 21.8, 24.1, 32.0, 51.3, 57.8, 68.2, 58.0, 59.3, 52.0, 81.4, 74.8, 66.5, 61.6,
                                    82.9, 74.3, 74.2, 70.0, 61.3, 38.7])
    prompt_2_all_models_icon = np.array([52.7, 24.3, 25.7, 31.1, 51.4, 64.3, 72.2, 58.9, 60.3, 49.5, 84.9, 77.4, 70.4,
                                         63.4, 79.8, 72.8, 76.3, 80.9, 68.4, 63.2])
    print("Avg. diff. between non-icon and icon accuracy for prompt 2:", prompt_2_all_models.mean() -
          prompt_2_all_models_icon.mean())


def analyze_summary():
    with open("../results/analysis/results/summary_v2.json", "r") as file:
        summary = json.load(file)

    for model, results in summary.items():
        for prompt in [f"prompt_{p}" for p in range(1, 5)]:
            if model == "mistral" and prompt in ["prompt_1", "prompt_2"]:
                continue

            if not model.startswith("belief_graphs") and model != "clip":
                text_result = np.array(results[prompt]["text_acc"])
                icon_result = np.array(results[prompt]["icon_acc"])

                summary[model][prompt]["text_mean"] = text_result.mean()
                summary[model][prompt]["text_sd"] = text_result.std()
                summary[model][prompt]["icon_mean"] = icon_result.mean()
                summary[model][prompt]["icon_sd"] = icon_result.std()

    for model in ["belief_graphs_gpt-40", "belief_graphs_gpt-4o-mini", "clip"]:
        results = summary[model]
        text_result = np.array(results["text_acc"])
        icon_result = np.array(results["icon_acc"])

        summary[model]["text_mean"] = text_result.mean()
        summary[model]["text_sd"] = text_result.std()
        summary[model]["icon_mean"] = icon_result.mean()
        summary[model]["icon_sd"] = icon_result.std()

    with open("../results/analysis/results/summary.json", "w") as file:
        json.dump(summary, file, indent=3)


def remove_faulty_puzzles():
    faulty_images = ["go_to_the_ends_of_the_earth_1_non-icon.png", "microchips_1.png", "midgame_1.png"]
    for file in Path("../results/analysis/results").rglob("*clip.json"):
        if "backup" not in str(file) and "human" not in str(file) and "closed_source" not in str(file):
            with open(file, "r") as results:
                results = json.load(results)
            print(len(results["results"]))
            print(file)
            for i, result in enumerate(results["results"]):
                if os.path.basename(result["image"]) in faulty_images:
                    # del results["results"][i]
                    pass

            # with open(file, "w") as file_:
            #     json.dump(results, file_, indent=3)


def extract_closed_source_model_results():
    models = ["gpt-4o", "gpt-4o-mini", "gemini-pro", "gemini-flash"]
    runs = [f"run_{i}" for i in [2, 3]]
    prompts = [f"prompt_{i}" for i in [1, 2, 3, 4]]
    faulty_puzzles_id = [501, 676, 689]

    prompt_template = ("I have the following text:\n\"{}\"\n\nPlease extract the answer being given in this text. "
                       "Remember that the answer can also refer to any of the symbols as well (either A, B, C, D). "
                       "Respond with 'None' if the text doesn't sufficiently match any of the options. "
                       "Respond with a comma-separated list of answers if you think there is more than one suitable answer. "
                       "Respond with only these options: {}")

    for model, run, prompt in tqdm(list(itertools.product(*[models, runs, prompts])), desc="Extracting responses..."):
        path = f"../results/analysis/results/{run}/{prompt}/{model}"
        for filename in glob.glob(f"{path}/*.json"):
            puzzle_id = int(os.path.basename(filename).split(".")[0])
            if puzzle_id in faulty_puzzles_id:
                continue
            with open(filename, "r") as file:
                result = json.load(file)
            prompt = result["prompt"]
            response = result["gpt4v_response"] if "gpt4v_response" in result else result["gemini_pro_response"]
            options = re.findall(r"\([A-Z]\)\s(.*?)(?=\s\([A-Z]\)|$)", prompt)
            options = " ".join(f"{symbol}) {option}" for symbol, option in zip(["A", "B", "C", "D"], options))
            extraction_prompt = prompt_template.format(*[response, options])
            extracted_response = make_safe_prompt(extraction_prompt)
            if extracted_response == "None":
                extracted_response = response
            result["extracted_response"] = extracted_response



def analyze_closed_source_model_results():
    models = ["gpt-4o", "gpt-4o-mini", "gemini-pro", "gemini-flash"]
    runs = [2, 3]
    prompts = [1, 2, 3, 4]
    faulty_puzzles_id = [501, 676, 689]

    for model in models:
        for run in [f"run_{i}" for i in runs]:
            for prompt in [f"prompt_{i}" for i in prompts]:
                path = f"../results/analysis/results/{run}/{prompt}/{model}"
                for filename in glob.glob(f"{path}/*.json"):
                    puzzle_id = int(os.path.basename(filename).split(".")[0])
                    if puzzle_id in faulty_puzzles_id:
                        continue


if __name__ == "__main__":
    # analyze_new_results()
    # analyze_summary()
    # remove_faulty_puzzles()
    #
    # with open("../benchmark.json", "r") as file:
    #     benchmark = json.load(file)
    #     print(len(benchmark))

    # analyze_closed_source_model_results()
    extract_closed_source_model_results()
