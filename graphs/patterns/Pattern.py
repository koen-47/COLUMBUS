import json

import inflect
import matplotlib.colors as mcolors

with open("./saved/homophones.json", "r") as file:
    homophones = json.load(file)


class Pattern:
    class Structural:
        class Position:
            HIGH = ["high", "up"]
            RIGHT = ["right"]

        class Repetition:
            REPETITION_TWO = ["two"]
            REPETITION_FOUR = ["four"]

    class Individual:
        COLOR = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow"]
        REVERSE = ["reverse", "back", "mirror", "inverse", "rear"]
        CROSS = ["cross"]

    ALL_KEYWORDS = {
        "color": Individual.COLOR,
        "reverse": Individual.REVERSE,
        "cross": Individual.CROSS,
        "position_high": Structural.Position.HIGH,
        "position_right": Structural.Position.RIGHT,
        "repetition_two": Structural.Repetition.REPETITION_TWO,
        "repetition_four": Structural.Repetition.REPETITION_FOUR,
    }

    ALL_RULES = ["color", "reverse", "cross", "high", "repeat", "position", "sound"]

    @staticmethod
    def find_all(word):
        homophones_overlap = {}
        if word in homophones:
            homophones_ = set(homophones[word]["perfect"]).union(set(homophones[word]["close"]))
            homophones_overlap = {rule: list(homophones_.intersection(set(keyword))) for rule, keyword in
                                  Pattern.ALL_KEYWORDS.items() if len(homophones_.intersection(set(keyword))) > 0}

        patterns = {"template": "base"}
        if word in Pattern.Individual.COLOR:
            patterns["color"] = word
        if word in Pattern.Individual.REVERSE:
            patterns["reverse"] = True
        if word in Pattern.Individual.CROSS:
            patterns["cross"] = True
        if word in Pattern.Structural.Position.HIGH:
            patterns["position"] = "high"
        if word in Pattern.Structural.Position.RIGHT:
            patterns["position"] = "right"
        if word in Pattern.Structural.Repetition.REPETITION_TWO:
            patterns["repeat"] = 2
        if word in Pattern.Structural.Repetition.REPETITION_FOUR:
            patterns["repeat"] = 4
        if "repetition_four" in homophones_overlap:
            patterns["repeat"] = 4
            patterns["sound"] = {word: homophones_overlap["repetition_four"]}
        if "repetition_two" in homophones_overlap:
            patterns["repeat"] = 2
            patterns["sound"] = {word: homophones_overlap["repetition_two"]}
        return patterns
