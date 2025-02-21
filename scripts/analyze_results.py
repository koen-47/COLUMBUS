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

    prompt_2 = np.array([56.15, 24.74, 23.95, 32.02, 51.47, 58.02, 68.24, 58.02, 59.28, 66.82, 80.89, 73.96, 71.56, 64.42,
                         81.28, 73.53, 69.98, 72.0, 64.37, 45.93])
    prompt_2_icon = np.array([52.56, 24.08, 25.61, 31.00, 51.75, 63.16, 71.97, 58.76, 60.11, 73.13, 83.34, 77.69, 77.52, 67.44,
                              79.2, 74.36, 72.1, 75.88, 71.67, 60.0])
    human, human_icon = 98.00, 93.21

    print("Analysis of overall performance (Table 2)")
    print(f"Avg. prompt 2:", prompt_2.mean())
    print(f"Avg. prompt 2 (icon):", prompt_2_icon.mean())
    print(f"Diff. human vs. prompt 2:", np.abs(prompt_2.mean() - human))
    print(f"Diff. human vs. prompt 2 (icon):", np.abs(prompt_2_icon.mean() - human_icon))

    prompt_1_four_models = np.array([75.81, 64.84, 32.44])
    prompt_1_four_models_icon = np.array([76.41, 67.12, 32.61])

    prompt_2_four_models = np.array([80.87, 68.24, 32.02])
    prompt_2_four_models_icon = np.array([83.34, 71.97, 31.0])

    prompt_3_four_models = np.array([90.55, 85.07, 42.93, 70.61])
    prompt_3_four_models_icon = np.array([93.63, 91.64, 47.44, 75.29])

    prompt_4_four_models = np.array([90.86, 84.91, 45.55, 73.71])
    prompt_4_four_models_icon = np.array([93.24, 90.84, 49.33, 76.82])

    print("\nAnalysis of the four models from each category (Figure 5)")
    print(f"(four models) Diff. prompt 1 vs. prompt 2 (no icon):",
          prompt_2_four_models.mean() - prompt_1_four_models.mean())
    print(f"(four models) Diff. prompt 1 vs. prompt 2 (icon):",
          prompt_2_four_models_icon.mean() - prompt_1_four_models_icon.mean())

    print(f"(four models) Diff. prompt 2 vs. prompt 3 (no icon):",
          prompt_3_four_models.mean() - prompt_2_four_models.mean())
    print(f"(four models) Diff. prompt 2 vs. prompt 3 (icon):",
          prompt_3_four_models_icon.mean() - prompt_2_four_models_icon.mean())

    print(f"(four models) Diff. prompt 3 vs. prompt 4 (no icon):",
          prompt_4_four_models.mean() - prompt_3_four_models.mean())
    print(f"(four models) Diff. prompt 3 vs. prompt 4 (icon):",
          prompt_4_four_models_icon.mean() - prompt_3_four_models_icon.mean())

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


if __name__ == "__main__":
    analyze_new_results()

