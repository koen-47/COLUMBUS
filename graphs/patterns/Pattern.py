import json

import inflect
import matplotlib.colors as mcolors

from ..templates.Template import Template

inflect = inflect.engine()

with open("./saved/homophones.json", "r") as file:
    homophones = json.load(file)


class Pattern:
    class Structural:
        class Position:
            HIGH = ["high", "up"]
            RIGHT = ["right"]

        class Repetition:
            TWO = ["two"]
            FOUR = ["four"]

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
        "repetition_two": Structural.Repetition.TWO,
        "repetition_four": Structural.Repetition.FOUR,
    }

    ALL_RULES = ["color", "reverse", "cross", "high", "repeat", "position", "sound"]
    STRUCTURAL_RULES = [field.lower() for field in dir(Structural) if not field.startswith("__")]

    @staticmethod
    def find_all(word):
        homophones_overlap = {}
        if word in homophones:
            homophones_ = set(homophones[word]["perfect"]).union(set(homophones[word]["close"]))
            homophones_overlap = {rule: list(homophones_.intersection(set(keyword))) for rule, keyword in
                                  Pattern.ALL_KEYWORDS.items() if len(homophones_.intersection(set(keyword))) > 0}

        word_singular = inflect.singular_noun(word)
        if word_singular in homophones:
            homophones_ = set(homophones[word_singular]["perfect"]).union(set(homophones[word_singular]["close"]))
            homophones_overlap = {rule: list(homophones_.intersection(set(keyword))) for rule, keyword in
                                  Pattern.ALL_KEYWORDS.items() if len(homophones_.intersection(set(keyword))) > 0}

        patterns = {}
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
        if word in Pattern.Structural.Repetition.TWO:
            patterns["repeat"] = 2
        if word in Pattern.Structural.Repetition.FOUR:
            patterns["repeat"] = 4
        if "repetition_four" in homophones_overlap:
            patterns["repeat"] = 4
            patterns["sound"] = {word: homophones_overlap["repetition_four"]}
        if "repetition_two" in homophones_overlap:
            patterns["repeat"] = 2
            patterns["sound"] = {word: homophones_overlap["repetition_two"]}
        if "position_right" in homophones_overlap:
            patterns["position"] = "right"
            patterns["sound"] = {word: homophones_overlap["position_right"]}
        return patterns

