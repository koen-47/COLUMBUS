import json

import inflect
import matplotlib.colors as mcolors

with open("./saved/homophones.json", "r") as file:
    homophones = json.load(file)


class Pattern:
    class Unary:
        COLOR = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow"]
        REVERSE = ["reverse", "back", "mirror", "inverse", "rear"]
        CROSS = ["cross"]

        class Position:
            HIGH = ["high", "up"]
            RIGHT = ["right"]

        class Repetition:
            REPETITION_TWO = ["two"]
            REPETITION_FOUR = ["four"]

    ALL = {
        "color": Unary.COLOR,
        "reverse": Unary.REVERSE,
        "cross": Unary.CROSS,
        "high": Unary.Position.HIGH,
        "repetition_two": Unary.Repetition.REPETITION_TWO,
        "repetition_four": Unary.Repetition.REPETITION_FOUR,
    }

    @staticmethod
    def find_all(word, is_plural):
        homophones_overlap = {}
        if word in homophones:
            homophones_ = set(homophones[word]["perfect"]).union(set(homophones[word]["close"]))
            homophones_overlap = {rule: list(homophones_.intersection(set(keyword))) for rule, keyword in
                                  Pattern.ALL.items() if len(homophones_.intersection(set(keyword))) > 0}

        patterns = {"template": "base"}
        if word in Pattern.Unary.COLOR:
            patterns["color"] = word
        if word in Pattern.Unary.REVERSE:
            patterns["reverse"] = True
        if word in Pattern.Unary.CROSS:
            patterns["cross"] = True
        if word in Pattern.Unary.HIGH:
            patterns["template"] = "high"
        if word in Pattern.Unary.RIGHT:
            patterns["template"] = "right"
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

