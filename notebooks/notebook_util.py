import os
import json

import pandas as pd


def show_puzzles_3x3():
    pass

def load_inputs_in_columbus():
    compounds = pd.read_csv("../data/input/ladec_raw_small.csv")
    custom_compounds = pd.read_csv("../data/input/custom_compounds.csv")
    compounds = pd.concat([compounds, custom_compounds]).reset_index(drop=True)

    compounds_in_columbus = []
    phrases_in_columbus = []
    with open("../benchmark.json", "r") as file:
        benchmark = json.load(file)
        for puzzle in benchmark:
            answer = os.path.basename(puzzle["image"]).split(".")[0]
            answer_parts = answer.split("_")
            if answer.endswith("non-icon") or answer.endswith("icon"):
                answer_parts = answer_parts[:-1]
            if answer_parts[-1].isnumeric():
                answer_parts = answer_parts[:-1]
            if len(answer_parts) == 1:
                row = compounds.loc[compounds["stim"] == answer_parts[0]].values.flatten().tolist()
                compounds_in_columbus.append({
                    "word_1": row[0],
                    "word_2": row[1],
                    "compound": row[2],
                    "is_plural": row[3] == 1
                })
            elif len(answer_parts) > 1:
                answer = " ".join(answer_parts)
                phrases_in_columbus.append(answer)
                
    return compounds_in_columbus, phrases_in_columbus