import itertools
import json
import re

from statsmodels.stats.contingency_tables import mcnemar


def standardize_result(result):
    output = result["output"]
    if re.match(r"\([A-D]\)\s.+", output):
        letter = output.split()[0][1]
        answer = " ".join(output.split()[1:])
        is_correct = {letter: answer} == result["correct"]
        # print({letter: answer}, result["correct"])
        result["is_correct"] = is_correct
    elif re.match(r"[A-D]\)\s.+", output):
        letter = output.split()[0][0]
        answer = " ".join(output.split()[1:])
        is_correct = {letter: answer} == result["correct"]
        result["is_correct"] = is_correct
    elif re.match(r"^[A-D]$", output):
        letter = output
        answer = result["options"][letter]
        is_correct = {letter: answer} == result["correct"]
        result["is_correct"] = is_correct
    else:
        answer_to_letter = {v: k for k, v in result["options"].items()}
        if output in answer_to_letter:
            letter = answer_to_letter[output]
            is_correct = {letter: output} == result["correct"]
            result["is_correct"] = is_correct
        else:
            # print(output, result["options"], result["correct"])
            result["is_correct"] = False


def format_blip2_results(file_path):
    with open(file_path, "r") as file:
        results = json.load(file)
    for result in results:
        result = standardize_result(result)
    return results

def format_llava_results(file_path):
    with open(file_path, "r") as file:
        results = json.load(file)
    for result in results:
        result["output"] = re.split("ASSISTANT: ", result["output"])[1]
        result = standardize_result(result)
    return results


def analyze_compounds():
    blip2_7b_compounds = format_blip2_results("../results/experiments/blip-2_2.7b_compounds.json")
    blip6_7b_compounds = format_blip2_results("../results/experiments/blip-2_6.7b_compounds.json")
    llava_compounds = format_llava_results("../results/experiments/llava-1.5-13b_compounds.json")

    def compute_accuracy(results):
        n_correct = len([result for result in results if result["is_correct"] is True])
        return (n_correct / len(results)) * 100

    print("=== MODEL RESULTS (COMPOUNDS) ==")
    print(f"BLIP-2 2.7b: {compute_accuracy(blip2_7b_compounds):.2f}%")
    print(f"BLIP-2 6.7b: {compute_accuracy(blip6_7b_compounds):.2f}%")
    print(f"Llava-1.5-13b: {compute_accuracy(llava_compounds):.2f}%")

    combinations = list(itertools.permutations([
        {"BLIP-2 2.7b": blip2_7b_compounds},
        {"BLIP-2 6.7b": blip6_7b_compounds},
        {"Llava-1.5-13b": llava_compounds}
    ], 2))

    for model_1, model_2 in combinations:
        model_1_name, model_1_result = next(iter(model_1)), list(model_1.values())[0]
        model_2_name, model_2_result = next(iter(model_2)), list(model_2.values())[0]
        p_value = perform_mcnemar_test(model_1_result, model_2_result)
        print(f"McNemar test ({model_1_name} vs. {model_2_name}): {p_value:.3f}")


def analyze_phrases():
    blip2_7b_phrases = format_blip2_results("../results/experiments/blip-2_2.7b_phrases.json")
    blip6_7b_phrases = format_blip2_results("../results/experiments/blip-2_6.7b_phrases.json")
    llava_phrases = format_llava_results("../results/experiments/llava-1.5-13b_phrases.json")

    def compute_accuracy(results):
        n_correct = len([result for result in results if result["is_correct"] is True])
        return (n_correct / len(results)) * 100

    print("=== MODEL RESULTS (PHRASES) ===")
    print(f"BLIP-2 2.7b: {compute_accuracy(blip2_7b_phrases):.2f}%")
    print(f"BLIP-2 6.7b: {compute_accuracy(blip6_7b_phrases):.2f}%")
    print(f"Llava-1.5-13b: {compute_accuracy(llava_phrases):.2f}%")

    combinations = list(itertools.permutations([
        {"BLIP-2 2.7b": blip2_7b_phrases},
        {"BLIP-2 6.7b": blip6_7b_phrases},
        {"Llava-1.5-13b": llava_phrases}
    ], 2))

    for model_1, model_2 in combinations:
        model_1_name, model_1_result = next(iter(model_1)), list(model_1.values())[0]
        model_2_name, model_2_result = next(iter(model_2)), list(model_2.values())[0]
        p_value = perform_mcnemar_test(model_1_result, model_2_result)
        print(f"McNemar test ({model_1_name} vs. {model_2_name}): {p_value:.3f}")


def perform_mcnemar_test(results_1, results_2):
    a, b, c, d = 0, 0, 0, 0
    for result_1, result_2 in zip(results_1, results_2):
        is_correct_1, is_correct_2 = result_1["is_correct"], result_2["is_correct"]
        if is_correct_1 and is_correct_2:
            a += 1
        elif is_correct_1 and not is_correct_2:
            b += 1
        elif not is_correct_1 and is_correct_2:
            c += 1
        elif not is_correct_1 and not is_correct_2:
            d += 1
    table = [[a, b],
             [c, d]]

    mcnemar_result = mcnemar(table, exact=False, correction=True).pvalue
    return mcnemar_result


analyze_compounds()
print()
analyze_phrases()
