import json

import inflect
import matplotlib.colors as mcolors

from ..templates.Template import Template

inflect = inflect.engine()

with open("./saved/homophones.json", "r") as file:
    homophones = json.load(file)


class Pattern:
    class Relational:
        INSIDE = ["in", "inside"]
        OUTSIDE = ["out", "outside"]
        ABOVE = ["above", "over", "on", "upon"]
        NEXT_TO = ["next"]

    class Individual:
        COLOR = ["black", "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow"]
        CROSS = ["cross"]

        class Direction:
            UP = ["up"]
            DOWN = ["down"]
            REVERSE = ["reverse", "back", "mirror", "inverse", "rear"]

        class Style:
            class Size:
                BIG = ["big", "large"]
                SMALL = ["small", "little"]

        class Highlight:
            class Arrow:
                AFTER = ["after", "end"]
                BEFORE = ["before", "begin", "start"]
                MIDDLE = ["middle", "mid"]

        class Position:
            HIGH = ["high"]
            RIGHT = ["right"]
            LEFT = ["left"]
            LOW = ["low"]

        class Repetition:
            TWO = ["two", "double", "to"]
            FOUR = ["four",  "quarter"]

    ALL_KEYWORDS = {
        "color": Individual.COLOR,
        "reverse": Individual.Direction.REVERSE,
        "cross": Individual.CROSS,
        "position_high": Individual.Position.HIGH,
        "position_right": Individual.Position.RIGHT,
        "repetition_two": Individual.Repetition.TWO,
        "repetition_four": Individual.Repetition.FOUR,
    }

    ALL_RULES = ["color", "reverse", "cross", "high", "repeat", "position", "direction", "size", "sound", "highlight"]

    IGNORE = ["the", "a", "of", "is", "let", "my", "and"]

    @staticmethod
    def find_all(word, is_plural):
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
        # INDIVIDUAL PATTERNS
        if word in Pattern.Individual.COLOR:
            patterns["color"] = word
        if word in Pattern.Individual.CROSS or word_singular in Pattern.Individual.CROSS:
            patterns["cross"] = True
        if word in Pattern.Individual.Direction.UP or word_singular in Pattern.Individual.Direction.UP:
            patterns["direction"] = "up"
        if word in Pattern.Individual.Direction.DOWN or word_singular in Pattern.Individual.Direction.DOWN:
            patterns["direction"] = "down"
        if word in Pattern.Individual.Direction.REVERSE or word_singular in Pattern.Individual.Direction.REVERSE:
            patterns["direction"] = "reverse"
        if word in Pattern.Individual.Style.Size.BIG or word_singular in Pattern.Individual.Style.Size.BIG:
            patterns["size"] = "big"
        if word in Pattern.Individual.Style.Size.SMALL or word_singular in Pattern.Individual.Style.Size.SMALL:
            patterns["size"] = "small"
        if word in Pattern.Individual.Highlight.Arrow.AFTER or word_singular in Pattern.Individual.Highlight.Arrow.AFTER:
            patterns["highlight"] = "after"
        if word in Pattern.Individual.Highlight.Arrow.BEFORE or word_singular in Pattern.Individual.Highlight.Arrow.BEFORE:
            patterns["highlight"] = "before"
        if word in Pattern.Individual.Highlight.Arrow.MIDDLE or word_singular in Pattern.Individual.Highlight.Arrow.MIDDLE:
            patterns["highlight"] = "middle"

        # STRUCTURAL PATTERNS
        if word in Pattern.Individual.Position.HIGH:
            patterns["position"] = "high"
        if word in Pattern.Individual.Position.RIGHT:
            patterns["position"] = "right"
        if word in Pattern.Individual.Position.LEFT:
            patterns["position"] = "left"
        if word in Pattern.Individual.Position.LOW:
            patterns["position"] = "low"
        if not is_plural:
            patterns["repeat"] = 1
        if word in Pattern.Individual.Repetition.TWO or is_plural:
            patterns["repeat"] = 2
        if word in Pattern.Individual.Repetition.FOUR:
            patterns["repeat"] = 4

        # INCLUDE SOUND PATTERNS
        if "repetition_four" in homophones_overlap:
            patterns["repeat"] = 4
            patterns["sound"] = {word: homophones_overlap["repetition_four"]}
        if "repetition_two" in homophones_overlap:
            patterns["repeat"] = 2
            patterns["sound"] = {word: homophones_overlap["repetition_two"]}
        if "position_right" in homophones_overlap:
            patterns["position"] = "right"
            patterns["sound"] = {word: homophones_overlap["position_right"]}
        # print(word, patterns)
        return patterns

    @staticmethod
    def get_all_relational(as_dict=True):
        if not as_dict:
            return Pattern.Relational.INSIDE + Pattern.Relational.OUTSIDE + Pattern.Relational.ABOVE

        return {
            "inside": Pattern.Relational.INSIDE,
            "outside": Pattern.Relational.OUTSIDE,
            "above": Pattern.Relational.ABOVE
        }
