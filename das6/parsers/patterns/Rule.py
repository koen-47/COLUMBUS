import json
import os

import inflect

inflect = inflect.engine()


class Rule:
    class Relational:
        INSIDE = ["in", "inside", "into"]
        OUTSIDE = ["out", "outside"]
        ABOVE = ["above", "over", "on", "upon"]
        NEXT_TO = ["next"]

    class Individual:
        class Direction:
            UP = ["up"]
            DOWN = ["down"]
            REVERSE = ["reverse", "back", "mirror", "inverse", "rear", "left", "flip"]

        class Style:
            COLOR = ["black", "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow", "gold"]
            CROSS = ["cross", "crossed", "crossing"]

            class Size:
                BIG = ["big", "large", "grand", "bigger", "biggest", "jumbo", "giant"]
                SMALL = ["small", "little", "micro", "smaller", "smallest", "miniature"]

        class Highlight:
            AFTER = ["after", "end", "behind"]
            BEFORE = ["before", "begin", "start", "left", "starting", "beginning"]
            MIDDLE = ["middle", "mid", "my"]

        class Position:
            HIGH = ["high"]
            RIGHT = ["right"]
            LEFT = ["left"]
            LOW = ["low"]

        class Repetition:
            TWO = ["two", "double", "to"]
            FOUR = ["four"]

    ALL_RULES = ["color", "reverse", "cross", "high", "repeat", "position", "direction", "size", "sound", "highlight", "icon"]

    @staticmethod
    def find_all(word, is_plural):
        conflicts = [rule for rule, keyword in Rule.get_all_rules()["individual"].items() if word in keyword]
        word_singular = inflect.singular_noun(word) if word is not None else word
        rules = {}

        # INDIVIDUAL PATTERNS
        if word in Rule.Individual.Style.COLOR:
            rules["color"] = word
        if word in Rule.Individual.Style.CROSS or word_singular in Rule.Individual.Style.CROSS:
            rules["cross"] = True
        if word in Rule.Individual.Direction.UP or word_singular in Rule.Individual.Direction.UP:
            rules["direction"] = "up"
        if word in Rule.Individual.Direction.DOWN or word_singular in Rule.Individual.Direction.DOWN:
            rules["direction"] = "down"
        if word in Rule.Individual.Direction.REVERSE or word_singular in Rule.Individual.Direction.REVERSE:
            rules["direction"] = "reverse"
        if word in Rule.Individual.Style.Size.BIG or word_singular in Rule.Individual.Style.Size.BIG:
            rules["size"] = "big"
        if word in Rule.Individual.Style.Size.SMALL or word_singular in Rule.Individual.Style.Size.SMALL:
            rules["size"] = "small"
        if word in Rule.Individual.Highlight.AFTER or word_singular in Rule.Individual.Highlight.AFTER:
            rules["highlight"] = "after"
        if word in Rule.Individual.Highlight.BEFORE or word_singular in Rule.Individual.Highlight.BEFORE:
            rules["highlight"] = "before"
        if word in Rule.Individual.Highlight.MIDDLE or word_singular in Rule.Individual.Highlight.MIDDLE:
            rules["highlight"] = "middle"
        if word in Rule.Individual.Position.HIGH:
            rules["position"] = "high"
        if word in Rule.Individual.Position.RIGHT:
            rules["position"] = "right"
        if word in Rule.Individual.Position.LEFT:
            rules["position"] = "left"
        if word in Rule.Individual.Position.LOW:
            rules["position"] = "low"

        if not is_plural:
            rules["repeat"] = 1
        if word in Rule.Individual.Repetition.TWO or is_plural:
            rules["repeat"] = 2
        if word in Rule.Individual.Repetition.FOUR:
            rules["repeat"] = 4

        # INCLUDE SOUND PATTERNS
        with open(f"{os.path.dirname(__file__)}/../../saved/homophones_v2.json", "r") as file:
            homophones = json.load(file)

        if word not in homophones and word_singular not in homophones:
            return rules, conflicts

        if word in homophones:
            homophones = homophones[word]
        elif word_singular in homophones:
            homophones = homophones[word_singular]

        if "4" in homophones:
            rules["repeat"] = 4
            rules["sound"] = {word: homophones}
        if "2" in homophones:
            rules["repeat"] = 2
            rules["sound"] = {word: homophones}
        if "right" in homophones:
            rules["position"] = "right"
            rules["sound"] = {word: homophones}

        return rules, conflicts

    @staticmethod
    def get_all_rules():
        return {
            "relational": {
                "inside": Rule.Relational.INSIDE,
                "outside": Rule.Relational.OUTSIDE,
                "above": Rule.Relational.ABOVE,
                "next_to": Rule.Relational.NEXT_TO,
            },
            "individual": {
                "direction_up": Rule.Individual.Direction.UP,
                "direction_down": Rule.Individual.Direction.DOWN,
                "direction_reverse": Rule.Individual.Direction.REVERSE,
                "color": Rule.Individual.Style.COLOR,
                "cross": Rule.Individual.Style.CROSS,
                "size_big": Rule.Individual.Style.Size.BIG,
                "size_small": Rule.Individual.Style.Size.SMALL,
                "highlight_after": Rule.Individual.Highlight.AFTER,
                "highlight_middle": Rule.Individual.Highlight.MIDDLE,
                "highlight_before": Rule.Individual.Highlight.BEFORE,
                "position_high": Rule.Individual.Position.HIGH,
                "position_low": Rule.Individual.Position.LOW,
                "position_left": Rule.Individual.Position.LEFT,
                "position_right": Rule.Individual.Position.RIGHT,
                "repetition_two": Rule.Individual.Repetition.TWO,
                "repetition_four": Rule.Individual.Repetition.FOUR
            }
        }
