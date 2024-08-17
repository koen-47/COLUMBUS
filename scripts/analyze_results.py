import itertools
import json
import re

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import statsmodels.api as sm
import pylab


def analyze_new_results():
    prompt_1 = np.array([24.8, 22.4, 32.6, 50.6, 57.8, 64.9, 63.2, 59.9, 49.2, 76.4, 69.0, 65.2, 60.5])
    prompt_2 = np.array([56.3, 21.8, 24.1, 32.0, 51.3, 57.8, 68.2, 58.0, 59.3, 52.0, 81.4, 74.8, 66.5, 61.6])
    prompt_3 = np.array([18.8, 22.9, 43.0, 42.8, 68.0, 85.1, 71.0, 63.5, 72.3, 90.7, 88.9, 83.8, 79.2])
    prompt_4 = np.array([21.5, 23.8, 45.6, 41.9, 68.2, 85.0, 71.2, 64.0, 76.7, 60.3, 91.0, 86.7, 84.8, 81.3])

    prompt_1_icon = np.array([23.0, 26.0, 25.0, 51.6, 68.9, 67.0, 61.4, 59.5, 51.9, 77.6, 66.0, 70.5, 63.9])
    prompt_2_icon = np.array([24.3, 25.7, 22.2, 51.4, 64.3, 72.2, 58.9, 60.3, 49.5, 84.9, 77.4, 70.4, 63.4])
    prompt_3_icon = np.array([17.0, 22.4, 47.3, 38.4, 73.2, 91.6, 69.4, 64.1, 76.8, 94.4, 88.0, 89.3, 81.3])
    prompt_4_icon = np.array([21.4, 21.1, 49.2, 39.2, 72.7, 90.8, 69.2, 64.9, 77.6, 85.3, 93.5, 88.4, 89.8, 81.5])

    prompt_2_small = np.array([21.8, 23.8, 57.8, 68.2, 58.0, 59.3, 52.0, 81.4, 74.8, 66.5, 61.6])
    prompt_2_small_icon = np.array([24.3, 22.2, 64.3, 72.2, 58.9, 60.3, 49.5, 84.9, 77.4, 70.4, 63.4])

    human, human_icon = 99.04, 91.51

    def calculate_avg_diff(arr_1, arr_2):
        return np.array([np.absolute(x1 - x2) for x1, x2 in zip(arr_1, arr_2)]).mean()

    def generate_qqplot(arr):
        sm.qqplot(arr, line="45")
        pylab.show()

    avg_diff_prompt_1_2_small = calculate_avg_diff(prompt_1, prompt_2_small)
    avg_diff_prompt_2_3 = calculate_avg_diff(prompt_2, prompt_3)
    avg_diff_prompt_2_3_small = calculate_avg_diff(prompt_2_small, prompt_3)
    avg_diff_prompt_3_4 = calculate_avg_diff(prompt_3, prompt_4)

    avg_diff_prompt_1_2_small_icon = calculate_avg_diff(prompt_1_icon, prompt_2_small_icon)
    avg_diff_prompt_2_3_icon = calculate_avg_diff(prompt_2_icon, prompt_3_icon)
    avg_diff_prompt_2_3_small_icon = calculate_avg_diff(prompt_2_small_icon, prompt_3_icon)
    avg_diff_prompt_3_4_icon = calculate_avg_diff(prompt_3_icon, prompt_4_icon)
    avg_diff_prompt_2_2_icon = calculate_avg_diff(prompt_2, prompt_2_icon)

    avg_diff_prompt_3_3_icon = calculate_avg_diff(prompt_3, prompt_3_icon)
    avg_diff_prompt_4_4_icon = calculate_avg_diff(prompt_4, prompt_4_icon)

    print(f"Avg. prompt 2:", prompt_2.mean())
    print(f"Avg. prompt 2:", prompt_2_icon.mean())
    print(f"Diff. human vs. prompt 2:", np.abs(prompt_2.mean() - human))
    print(f"Diff. human vs. prompt 2 (icon):", np.abs(prompt_2_icon.mean() - human_icon))
    print(f"Diff. prompt 1 and 2 (small):", avg_diff_prompt_1_2_small, avg_diff_prompt_1_2_small_icon)
    print(f"Diff. prompt 2 vs. prompt 2 icon:", avg_diff_prompt_2_2_icon)
    print(f"Diff. prompt 2 and 3 (small):", avg_diff_prompt_2_3_small, avg_diff_prompt_2_3_small_icon)
    print(f"Diff. prompt 2 and 3:", avg_diff_prompt_2_3, avg_diff_prompt_2_3_icon)
    print(f"Diff. prompt 3 and 4:", avg_diff_prompt_3_4, avg_diff_prompt_3_4_icon)
    print(f"Diff. prompt 3 (icon vs. no icon):", avg_diff_prompt_3_3_icon)
    print(f"Diff. prompt 4 (icon vs. no icon):", avg_diff_prompt_4_4_icon)
    print(f"Diff. prompt 2 and 3 (icon vs. no icon) {avg_diff_prompt_2_3 - avg_diff_prompt_2_3_icon}")

    individual_prompt_2 = np.array([43.18, 61.36, 44.17, 41.67, 56.14, 44.87, 55.09, 54.59, 65.12, 46.55])
    relational_prompt_2 = np.array([69.93, 59.94, 72.23, 72.39])
    modifier_prompt_2 = np.array([63.50, 64.40, 54.66])

    print("Non-icon statistics")
    print(individual_prompt_2.mean() - relational_prompt_2.mean())
    print(individual_prompt_2.mean() - modifier_prompt_2.mean())
    print(individual_prompt_2.mean())
    print(modifier_prompt_2.mean())

    individual_prompt_2_icon = np.array([42.50, 65.00, 52.50, 49.24, 53.33, 55.61, 53.09])
    relational_prompt_2_icon = np.array([63.21, 61.23, 61.92, 61.49])
    modifier_prompt_2_icon = np.array([62.44, 66.45, 68.15])

    print(f"Icon statistics")
    print(individual_prompt_2_icon.mean() - relational_prompt_2_icon.mean())
    print(individual_prompt_2_icon.mean() - modifier_prompt_2_icon.mean())

    print("Comparing non-icon vs. icon")
    print(np.array([prompt_1.mean(), prompt_2.mean(), prompt_3.mean(), prompt_4.mean()]).mean() -
          np.array([prompt_1_icon.mean(), prompt_2_icon.mean(), prompt_3_icon.mean(), prompt_4_icon.mean()]).mean())

    prompt_1_four_models = np.array([32.6, 64.9, 76.4])
    prompt_1_four_models_icon = np.array([32.4, 72.2, 84.9])

    prompt_2_four_models = np.array([31.97, 68.2, 81.4])
    prompt_2_four_models_icon = np.array([31.08, 72.2, 84.9])

    prompt_3_four_models = np.array([43.0, 85.1, 90.7, 72.3])
    prompt_3_four_models_icon = np.array([47.3, 91.6, 94.4, 76.8])

    print(f"(four models) Diff. prompt 1 vs. prompt 2 (no icon):", prompt_2_four_models.mean() - prompt_1_four_models.mean())
    print(f"(four models) Diff. prompt 1 vs. prompt 2 (icon):", prompt_2_four_models_icon.mean() - prompt_1_four_models_icon.mean())

    print(f"(four models) Diff. prompt 2 vs. prompt 3 (no icon):", prompt_3_four_models.mean() - prompt_2_four_models.mean())
    print(f"(four models) Diff. prompt 2 vs. prompt 3 (icon):", prompt_3_four_models_icon.mean() - prompt_2_four_models_icon.mean())

    gpt4o_individual_prompt_2 = np.array([43.48, 78.95, 57.69, 72.73, 80.77, 70.0, 72.22, 86.96, 85.19, 66.67])
    gpt4o_relational_prompt_2 = np.array([84.64, 89.34, 90.72, 93.1])
    gpt4o_modifier_prompt_2 = np.array([78.2, 75.92, 69.35])

    gpt4o_individual_prompt_2_icon = np.array([30.0, 100.0, 80.0, 54.55, 70.0, 100.0, 83.33])
    gpt4o_relational_prompt_2_icon = np.array([82.3, 81.73, 86.84, 88.89])
    gpt4o_modifier_prompt_2_icon = np.array([81.33, 83.12, 82.14])

    print(gpt4o_individual_prompt_2.mean() - gpt4o_relational_prompt_2.mean())
    print(gpt4o_individual_prompt_2.mean() - gpt4o_modifier_prompt_2.mean())

    print(gpt4o_individual_prompt_2_icon.mean() - gpt4o_relational_prompt_2_icon.mean())
    print(gpt4o_individual_prompt_2_icon.mean() - gpt4o_modifier_prompt_2_icon.mean())

    prompt_2_all_models = np.array([56.3, 21.8, 24.1, 32.0, 51.3, 57.8, 68.2, 58.0, 59.3, 52.0, 81.4, 74.8, 66.5, 61.6,
                                    82.9, 74.3, 74.2, 70.0, 61.3, 38.7])
    prompt_2_all_models_icon = np.array([52.7, 24.3, 25.7, 31.1, 51.4, 64.3, 72.2, 58.9, 60.3, 49.5, 84.9, 77.4, 70.4,
                                         63.4, 79.8, 72.8, 76.3, 80.9, 68.4, 63.2])
    print(prompt_2_all_models.mean() - prompt_2_all_models_icon.mean())


analyze_new_results()
