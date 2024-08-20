import numpy as np


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
