import json

import inflect
import matplotlib.colors as mcolors

with open("./saved/homophones.json", "r") as file:
    homophones = json.load(file)


class Pattern:
    ALL = ["color", "reverse", "cross", "high", "repetition_two", "repetition_four"]

    class Unary:
        # COLOR = [color.replace("tab:", "") for color in list(mcolors.TABLEAU_COLORS.keys())]
        # COLOR = list(mcolors.CSS4_COLORS.keys())
        COLOR = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow"]
        REVERSE = ["reverse", "back", "mirror", "inverse", "rear"]
        CROSS = ["cross"]
        HIGH = ["high", "up"]
        REPETITION_TWO = ["two"]
        REPETITION_FOUR = ["four"]

    ALL_KEYWORDS = {
        "color": Unary.COLOR,
        "reverse": Unary.REVERSE,
        "cross": Unary.CROSS,
        "high": Unary.HIGH,
        "repetition_two": Unary.REPETITION_TWO,
        "repetition_four": Unary.REPETITION_FOUR,
    }

    @staticmethod
    def find_all(word, is_plural):
        homophones_overlap = {}
        if word in homophones:
            homophones_ = set(homophones[word]["perfect"]).union(set(homophones[word]["close"]))
            homophones_overlap = {rule: list(homophones_.intersection(set(keyword))) for rule, keyword in
                                  Pattern.ALL_KEYWORDS.items() if len(homophones_.intersection(set(keyword))) > 0}

        patterns = {"template": "base", "is_plural": is_plural}
        # patterns = {"template": "base"}
        if word in Pattern.Unary.COLOR:
            patterns["color"] = word
        if word in Pattern.Unary.REVERSE:
            patterns["reverse"] = True
        if word in Pattern.Unary.CROSS:
            patterns["cross"] = True
        if word in Pattern.Unary.HIGH:
            patterns["template"] = "high"
        if word in Pattern.Unary.REPETITION_TWO:
            patterns["template"] = "repetition_two"
        if word in Pattern.Unary.REPETITION_FOUR:
            patterns["template"] = "repetition_four"
        if "repetition_four" in homophones_overlap:
            patterns["template"] = "repetition_four"
            patterns["sound"] = {word: homophones_overlap["repetition_four"]}
        if "repetition_two" in homophones_overlap:
            patterns["template"] = "repetition_two"
            patterns["sound"] = {word: homophones_overlap["repetition_two"]}
        return patterns

