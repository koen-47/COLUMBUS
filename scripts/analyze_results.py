import itertools
import json
import re

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import statsmodels.api as sm
import pylab


def analyze_new_results():
    prompt_1 = np.array([18.0, 24.67, 54.67, 29.0, 45.33, 25.67, 51.33, 55.33])
    prompt_2 = np.array([49.33, 21.33, 24.33, 58.33, 29.67, 44.0, 45.67, 48.0, 50.67, 50.0])
    prompt_3 = np.array([25.33, 24.67, 80.67, 38.33, 48.0, 57.0, 59.0, 64.00])
    prompt_4 = np.array([25.0, 24.0, 80.0, 36.67, 47.0, 56.33, 59.67, 63.00])

    prompt_1_icon = np.array([23.0, 21.45, 67.18, 34.11, 54.26, 40.83, 57.36, 60.98])
    prompt_2_icon = np.array([52.71, 24.03, 23.36, 73.13, 35.4, 51.94, 56.07, 58.4, 54.01, 68.99])
    prompt_3_icon = np.array([21.45, 23.26, 90.18, 42.89, 56.33, 62.79, 65.63, 69.25])
    prompt_4_icon = np.array([24.55, 24.29, 90.96, 41.6, 49.35, 58.91, 62.53, 70.28])

    prompt_2_small = np.array([21.33, 24.33, 58.33, 29.67, 44.0, 45.67, 48.0, 50.67])
    prompt_2_small_icon = np.array([24.03, 23.36, 73.13, 35.4, 51.94, 56.07, 58.4, 54.01])

    print(prompt_2.mean())
    print(prompt_2_icon.mean())

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

    print(avg_diff_prompt_1_2_small, avg_diff_prompt_1_2_small_icon)
    print(f"Diff prompt 2 vs. prompt 2 icon:", avg_diff_prompt_2_2_icon)
    print(avg_diff_prompt_2_3_small, avg_diff_prompt_2_3_small_icon)
    print(avg_diff_prompt_2_3, avg_diff_prompt_2_3_icon)
    print(avg_diff_prompt_3_4, avg_diff_prompt_3_4_icon)
    print(f"icon presence: {avg_diff_prompt_3_3_icon}, {avg_diff_prompt_4_4_icon}")
    print(f"icon presence diff: {avg_diff_prompt_2_3 - avg_diff_prompt_2_3_icon}")

    # generate_qqplot(np.absolute(prompt_3 - prompt_1))
    # generate_qqplot(np.absolute(prompt_4 - prompt_3))
    # generate_qqplot(np.absolute(prompt_3_icon - prompt_1_icon))
    # generate_qqplot(np.absolute(prompt_4_icon - prompt_3_icon))

    print(wilcoxon(prompt_1, prompt_2_small))
    print(wilcoxon(prompt_1_icon, prompt_2_small_icon))
    print(wilcoxon(prompt_2_small, prompt_3))
    print(wilcoxon(prompt_2_small_icon, prompt_3_icon))
    print(wilcoxon(prompt_3, prompt_4))
    print(wilcoxon(prompt_3_icon, prompt_4_icon))
    print("Wilcoxon prompt 2 vs prompt 2 icon:", wilcoxon(prompt_2, prompt_2_icon))

    individual_prompt_1 = np.array([39.68, 52.86, 27.14, 45.06, 30.95, 46.82, 51.19, 30.77, 59.82, 38.35])
    individual_prompt_2 = np.array([36.51, 52.14, 23.57, 43.96, 51.19, 44.44, 46.43, 36.26, 62.50, 42.11])
    individual_prompt_3 = np.array([52.91, 65.71, 45.00, 47.25, 47.62, 41.27, 57.14, 39.56, 61.61, 53.38])
    individual_prompt_4 = np.array([56.08, 65.71, 41.43, 47.25, 46.43, 43.65, 55.95, 41.76, 63.39, 52.63])

    relational_prompt_1 = np.array([60.23, 44.72, 64.28, 61.43])
    relational_prompt_2 = np.array([62.16, 40.37, 60.99, 74.29])
    relational_prompt_3 = np.array([63.19, 55.28, 61.54, 60.00])
    relational_prompt_4 = np.array([61.52, 54.04, 54.40, 64.29])

    modifier_prompt_1 = np.array([52.44, 50.59, 45.64])
    modifier_prompt_2 = np.array([55.04, 51.64, 54.01])
    modifier_prompt_3 = np.array([57.31, 50.33, 53.66])
    modifier_prompt_4 = np.array([56.05, 50.06, 54.01])

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
