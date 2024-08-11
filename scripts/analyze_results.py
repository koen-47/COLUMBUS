import itertools
import json
import re

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import statsmodels.api as sm
import pylab


def analyze_new_results():
    prompt_1 = np.array([24.8, 22.4, 25.0, 50.6, 57.8, 64.9, 63.2, 59.9])
    prompt_2 = np.array([21.8, 24.1, 23.8, 51.3, 57.8, 68.2, 58.0, 59.3])
    prompt_3 = np.array([18.8, 22.9, 43.0, 42.8, 68.0, 85.1, 71.0, 64.0, 72.3])
    prompt_4 = np.array([21.5, 23.8, 45.6, 41.9, 68.2, 85.0, 71.2, 64.0, 76.7])

    prompt_1_icon = np.array([23.0, 26.0, 25.0, 51.6, 68.9, 67.0, 61.4, 59.5])
    prompt_2_icon = np.array([24.3, 25.7, 22.2, 51.4, 64.3, 72.2, 58.9, 60.3])
    prompt_3_icon = np.array([17.0, 22.4, 47.3, 38.4, 73.2, 91.6, 69.4, 64.1, 76.8])
    prompt_4_icon = np.array([21.4, 21.1, 49.2, 39.2, 72.7, 90.8, 69.2, 64.9, 77.6])

    prompt_2_small = np.array([21.8, 23.8, 57.8, 68.2, 58.0, 59.3])
    prompt_2_small_icon = np.array([24.3, 22.2, 64.3, 72.2, 58.9, 60.3])

    def calculate_avg_diff(arr_1, arr_2):
        return np.array([np.absolute(x1 - x2) for x1, x2 in zip(arr_1, arr_2)]).mean()

    def generate_qqplot(arr):
        sm.qqplot(arr, line="45")
        pylab.show()

    print(prompt_2.mean())
    print(prompt_2_icon.mean())

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

    print(f"Diff. prompt 1 and 2 (small):", avg_diff_prompt_1_2_small, avg_diff_prompt_1_2_small_icon)
    print(f"Diff. prompt 2 vs. prompt 2 icon:", avg_diff_prompt_2_2_icon)
    print(f"Diff. prompt 2 and 3 (small):", avg_diff_prompt_2_3_small, avg_diff_prompt_2_3_small_icon)
    print(f"Diff. prompt 2 and 3:", avg_diff_prompt_2_3, avg_diff_prompt_2_3_icon)
    print(f"Diff. prompt 3 and 4:", avg_diff_prompt_3_4, avg_diff_prompt_3_4_icon)
    print(f"Diff. prompt 3 (icon vs. no icon):", avg_diff_prompt_3_3_icon)
    print(f"Diff. prompt 4 (icon vs. no icon):", avg_diff_prompt_4_4_icon)
    print(f"Diff. prompt 2 and 3 (icon vs. no icon) {avg_diff_prompt_2_3 - avg_diff_prompt_2_3_icon}")

    # print(wilcoxon(prompt_1, prompt_2_small))
    # print(wilcoxon(prompt_1_icon, prompt_2_small_icon))
    # print(wilcoxon(prompt_2_small, prompt_3))
    # print(wilcoxon(prompt_2_small_icon, prompt_3_icon))
    # print(wilcoxon(prompt_3, prompt_4))
    # print(wilcoxon(prompt_3_icon, prompt_4_icon))
    # print("Wilcoxon prompt 2 vs prompt 2 icon:", wilcoxon(prompt_2, prompt_2_icon))

    individual_prompt_2 = np.array([36.51, 52.14, 23.57, 43.96, 51.19, 44.44, 46.43, 36.26, 62.50, 42.11])
    relational_prompt_2 = np.array([62.16, 40.37, 60.99, 74.29])
    modifier_prompt_2 = np.array([55.04, 51.64, 54.01])

    print("Non-icon statistics")
    # print(np.array([individual_prompt_1.mean() - relational_prompt_1.mean(),
    #                 individual_prompt_2.mean() - relational_prompt_2.mean(),
    #                 individual_prompt_3.mean() - relational_prompt_3.mean(),
    #                 individual_prompt_4.mean() - relational_prompt_4.mean()]).mean())
    #
    # print(np.array([individual_prompt_1.std(),
    #                 individual_prompt_2.std(),
    #                 individual_prompt_3.std(),
    #                 individual_prompt_4.std()]).mean())
    #
    # print(np.array([relational_prompt_1.std(),
    #                 relational_prompt_2.std(),
    #                 relational_prompt_3.std(),
    #                 relational_prompt_4.std()]).mean())
    #
    # print(np.array([individual_prompt_1, individual_prompt_2, individual_prompt_3, individual_prompt_4]).mean())
    # print(np.array([modifier_prompt_1, modifier_prompt_2, modifier_prompt_3, modifier_prompt_4]).mean())
    #
    # print(np.array([relational_prompt_3[0] - relational_prompt_4[0],
    #                 relational_prompt_3[1] - relational_prompt_4[1],
    #                 relational_prompt_3[2] - relational_prompt_4[2],
    #                 relational_prompt_3[3] - relational_prompt_4[3]]).mean())

    print(individual_prompt_2.mean() - relational_prompt_2.mean())
    print(individual_prompt_2.std())
    print(relational_prompt_2.std())

    print(individual_prompt_2.mean())
    print(modifier_prompt_2.mean())
    print(individual_prompt_2.mean() - modifier_prompt_2.mean())

    individual_prompt_1_icon = np.array([34.29, 60.00, 42.86, 40.26, 57.14, 52.14, 38.10])
    individual_prompt_2_icon = np.array([38.57, 61.43, 42.86, 46.75, 61.43, 46.15, 48.81])
    individual_prompt_3_icon = np.array([41.43, 60.00, 55.71, 45.45, 57.14, 56.04, 45.24])
    individual_prompt_4_icon = np.array([51.43, 62.86, 57.14, 58.57, 48.05, 53.85, 42.86])

    relational_prompt_1_icon = np.array([58.25, 61.32, 59.38, 42.86])
    relational_prompt_2_icon = np.array([61.11, 63.75, 66.27, 45.11])
    relational_prompt_3_icon = np.array([62.10, 63.75, 62.64, 43.61])
    relational_prompt_4_icon = np.array([60.17, 64.55, 61.79, 49.62])

    modifier_prompt_1_icon = np.array([56.87, 59.60, 56.68])
    modifier_prompt_2_icon = np.array([60.40, 65.03, 60.83])
    modifier_prompt_3_icon = np.array([61.12, 60.59, 65.44])
    modifier_prompt_4_icon = np.array([59.04, 59.60, 62.67])

    print(f"Icon statistics")
    # print(np.array([individual_prompt_1_icon.mean() - relational_prompt_1_icon.mean(),
    #                 individual_prompt_2_icon.mean() - relational_prompt_2_icon.mean(),
    #                 individual_prompt_3_icon.mean() - relational_prompt_3_icon.mean(),
    #                 individual_prompt_4_icon.mean() - relational_prompt_4_icon.mean()]).mean())
    #
    # print(np.array([individual_prompt_1_icon.std(),
    #                 individual_prompt_2_icon.std(),
    #                 individual_prompt_3_icon.std(),
    #                 individual_prompt_4_icon.std()]).mean())
    #
    # print(np.array([relational_prompt_1_icon.std(),
    #                 relational_prompt_2_icon.std(),
    #                 relational_prompt_3_icon.std(),
    #                 relational_prompt_4_icon.std()]).mean())
    #
    # print(np.array([individual_prompt_1_icon, individual_prompt_2_icon, individual_prompt_3_icon, individual_prompt_4_icon]).mean())
    # print(np.array([modifier_prompt_1_icon, modifier_prompt_2_icon, modifier_prompt_3_icon, modifier_prompt_4_icon]).mean())
    #
    # print(np.array([relational_prompt_3_icon[0] - relational_prompt_4_icon[0],
    #                 relational_prompt_3_icon[1] - relational_prompt_4_icon[1],
    #                 relational_prompt_3_icon[2] - relational_prompt_4_icon[2],
    #                 relational_prompt_3_icon[3] - relational_prompt_4_icon[3]]).mean())

    print(individual_prompt_2_icon.mean() - relational_prompt_2_icon.mean())
    print(individual_prompt_2_icon.std())
    print(relational_prompt_2_icon.std())
    print(individual_prompt_2_icon.mean())
    print(individual_prompt_2_icon.mean() - modifier_prompt_2_icon.mean())

    print("Comparing non-icon vs. icon")
    print(np.array([prompt_1.mean(), prompt_2.mean(), prompt_3.mean(), prompt_4.mean()]).mean() -
          np.array([prompt_1_icon.mean(), prompt_2_icon.mean(), prompt_3_icon.mean(), prompt_4_icon.mean()]).mean())

analyze_new_results()
