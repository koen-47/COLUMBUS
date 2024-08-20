import json
import os

import inflect

inflect = inflect.engine()


class Rule:
    """
    Class that contains the hierarchy of rules used.
    """
    class Relational:
        """
        Class that contains the relational rule keywords.
        """
        INSIDE = ["in", "inside", "into"]
        OUTSIDE = ["out", "outside"]
        ABOVE = ["above", "over", "on", "upon"]
        NEXT_TO = ["next"]

    class Individual:
        """
        Class the contains the hierarchy for the individual rules.
        """
        class Direction:
            """
            Class that contains the direction rule keywords.
            """
            UP = ["up"]
            DOWN = ["down"]
            REVERSE = ["reverse", "back", "mirror", "inverse", "rear", "left", "flip"]

        class Style:
            """
            Class that contains the style rule keywords.
            """
            COLOR = ["black", "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "yellow", "gold"]
            CROSS = ["cross", "crossed", "crossing"]

            class Size:
                """
                Class that contains the size rule keywords.
                """
                BIG = ["big", "large", "grand", "bigger", "biggest", "jumbo", "giant"]
                SMALL = ["small", "little", "micro", "smaller", "smallest", "miniature"]

        class Highlight:
            """
            Class that contains the highlight rule keywords.
            """
            AFTER = ["after", "end", "behind"]
            BEFORE = ["before", "begin", "start", "left", "starting", "beginning"]
            MIDDLE = ["middle", "mid", "my"]

        class Repetition:
            """
            Class that contains the repetition rule keywords.
            """
            TWO = ["two", "double", "to"]
            FOUR = ["four"]

    ALL_RULES = ["color", "reverse", "cross", "high", "repeat", "position", "direction", "size", "sound", "highlight", "icon"]

    @staticmethod
    def find_all(word, is_plural):
        """
        Finds all the rules triggered by the specified word, based on the keywords defined above.

        :param word: specified word to match against the keywords defined above.
        :param is_plural: flag to denote if the specified word is plural or not.
        :return: dictionary mapping each rule category to its corresponding value (e.g., direction: up, color: red).
        """
        conflicts = [rule for rule, keyword in Rule.get_all_rules()["individual"].items() if word in keyword]

        # Singular version of the specified word
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

        # Set repetition rules based on the plurality of the specified word
        if not is_plural:
            rules["repeat"] = 1
        if word in Rule.Individual.Repetition.TWO or is_plural:
            rules["repeat"] = 2
        if word in Rule.Individual.Repetition.FOUR:
            rules["repeat"] = 4

        # INCLUDE SOUND PATTERNS
        with open(f"{os.path.dirname(__file__)}/../../data/misc/homophones_v2.json", "r") as file:
            homophones = json.load(file)

        # If the word is not a homophone, then don't proceed further with anything sound related
        if word not in homophones and word_singular not in homophones:
            return rules, conflicts

        # Check if the word + singular version of the word is a homophone
        if word in homophones:
            homophones = homophones[word]
        elif word_singular in homophones:
            homophones = homophones[word_singular]

        # Change the repetition rule value based on if it is a homophone
        if "4" in homophones:
            rules["repeat"] = 4
            rules["sound"] = {word: homophones}
        if "2" in homophones:
            rules["repeat"] = 2
            rules["sound"] = {word: homophones}

        return rules, conflicts

    @staticmethod
    def get_all_rules():
        """
        Gets the hierarchy of all rules as a dictionary.
        :return: hierarchy of rules as a dictionary.
        """
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
                "repetition_two": Rule.Individual.Repetition.TWO,
                "repetition_four": Rule.Individual.Repetition.FOUR
            }
        }
