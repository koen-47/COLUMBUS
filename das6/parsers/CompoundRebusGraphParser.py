import json
import os
import ast

import inflect

from parsers.patterns.Rule import Rule
from parsers.RebusGraph import RebusGraph
from util import remove_duplicate_graphs

inflect = inflect.engine()


class CompoundRebusGraphParser:
    def __init__(self):
        # Load homophones
        with open(f"{os.path.dirname(__file__)}/../data/misc/homophones_v2.json", "r") as file:
            self._homophones = json.load(file)
        # Load icons
        with open(f"{os.path.dirname(__file__)}/../data/misc/icons_v2.json", "r") as file:
            self._icons = {ast.literal_eval(labels): icon for labels, icon in json.load(file).items()}

    def parse(self, c1, c2, is_plural):
        # Check for patterns for either constituent word
        patterns_c1, conflicts_c1 = Rule.find_all(c1, is_plural)
        patterns_c2, conflicts_c2 = Rule.find_all(c2, is_plural)

        # Format patterns such that all mutually exclusive rules are handled individually
        if len(conflicts_c1) > 1:
            patterns_c1 = [{key: value for key, value in patterns_c1.items() if key not in list([conflict_.split("_")[0] for conflict_ in set(conflicts_c1) - {conflict}])} for conflict in conflicts_c1]
        if len(conflicts_c2) > 1:
            patterns_c2 = [{key: value for key, value in patterns_c2.items() if key not in list([conflict_.split("_")[0] for conflict_ in set(conflicts_c2) - {conflict}])} for conflict in conflicts_c2]

        # List to hold all possible generated puzzles
        graphs = []

        # Generate puzzles by combining both words into one
        for pattern_c1 in [patterns_c1] if type(patterns_c1) is not list else patterns_c1:
            graphs += self._generate_rebus(c2, pattern_c1, is_plural)
        for pattern_c2 in [patterns_c2] if type(patterns_c2) is not list else patterns_c2:
            graphs += self._generate_rebus(c1, pattern_c2, is_plural)

        # Generate puzzles by placing both words next to each other
        graphs += self._generate_rebus(c1, {}, is_plural, c2)

        # Remove duplicate puzzles
        graphs = remove_duplicate_graphs(graphs)

        for graph in graphs:
            graph.graph["answer"] = f"{c1}{c2}"

        return graphs

    def _generate_rebus(self, word_1, rules, is_plural, word_2=None):
        # Create RebusGraph object to store the rebus puzzle representation
        graph = RebusGraph()

        # Only generate the graph if there is more than 1 rule (repeat rule is always there)
        if len(rules) > 1:
            text, rules = self._parse_text(word_1, rules, is_plural)
            if len(text) > 1:
                graph_1, graph_2 = RebusGraph(), RebusGraph()
                graph_1_attrs = {rule: value for rule, value in rules.copy().items() if rule != "icon"}
                graph_2_attrs = rules.copy()
                del graph_2_attrs["sound"][list(rules["icon"].keys())[0]]
                graph_1.add_node(1, text=text[0], **graph_1_attrs)
                graph_2.add_node(1, text=text[1], **graph_2_attrs)
                return [graph_1, graph_2]
            graph.add_node(1, text=text[0], **rules)
            return [graph]

        # Generate graph by putting the two constituent words next to each other
        if word_2 is not None:
            word_1_homophone = self.parse_homophones(word_1)
            word_2_homophone = self.parse_homophones(word_2)
            word_1_icon = self.parse_icon(word_1)
            word_2_icon = self.parse_icon(word_2)
            node_1_attrs = {"text": word_1.upper(), "repeat": 1}
            node_2_attrs = {"text": word_2.upper(), "repeat": 1}

            if word_1_homophone != word_1 or word_2_homophone != word_2:
                if word_1_homophone != word_1:
                    node_1_attrs["sound"] = {word_1: word_1_homophone}
                    node_1_attrs["text"] = word_1_homophone.upper()
                if word_2_homophone != word_2:
                    node_2_attrs["sound"] = {word_2: word_2_homophone}
                    node_2_attrs["text"] = word_2_homophone.upper()
            if word_1_icon != word_1 or word_2_icon != word_2:
                if word_1_icon != word_1:
                    node_1_attrs["icon"] = {word_1: word_1_icon}
                    node_1_attrs["text"] = word_1_icon.upper()
                if word_2_icon != word_2:
                    node_2_attrs["icon"] = {word_2: word_2_icon}
                    node_2_attrs["text"] = word_2_icon.upper()

            if node_1_attrs == {"text": word_1.upper(), "repeat": 1} and node_2_attrs == {"text": word_2.upper(), "repeat": 1}:
                return []
            if "sound" in node_1_attrs and "icon" in node_1_attrs:
                return self._resolve_sound_icon_conflict(1, node_1_attrs, node_2_attrs)
            if "sound" in node_2_attrs and "icon" in node_2_attrs:
                return self._resolve_sound_icon_conflict(2, node_1_attrs, node_2_attrs)

            graph.add_node(1, **node_1_attrs)
            graph.add_node(2, **node_2_attrs)
            graph.add_edge(1, 2, rule="NEXT-TO")

            return [graph]

        # Return empty list if there is no more than 1 rule
        return []

    def parse_homophones(self, text):
        # Replace text with alternative that is phonetically identical
        if text in self._homophones:
            text = self._homophones[text][0]
            return text
        return text

    def parse_icon(self, text):
        for label, icon in self._icons.items():
            if text.lower() in label:
                text = icon
                return text
        return text

    def _parse_text(self, text, rules, is_plural):
        homophone = self.parse_homophones(text)
        icon = self.parse_icon(text)

        if text != homophone and text != icon:
            singular_text = inflect.singular_noun(homophone)
            if is_plural and singular_text is not False:
                if "sound" not in rules:
                    rules["sound"] = {}
                rules["sound"][text] = singular_text
                return [singular_text.upper(), icon], rules
            if "sound" not in rules:
                rules["sound"] = {}
            rules["sound"][text] = homophone
            rules["icon"] = {text: icon}
            return [homophone.upper(), icon], rules

        # Check for homophones
        if text != homophone:
            # Change text in case of plurality
            singular_text = inflect.singular_noun(homophone)
            if is_plural and singular_text is not False:
                if "sound" not in rules:
                    rules["sound"] = {}
                rules["sound"][text] = singular_text
                return [singular_text.upper()], rules
            if "sound" not in rules:
                rules["sound"] = {}
            rules["sound"][text] = homophone
            return [homophone.upper()], rules

        # Check for icons
        if text != icon:
            rules["icon"] = {text: icon}
            return [icon], rules

        # Return initial text if neither homophone nor icon is found
        singular_text = inflect.singular_noun(text)
        if is_plural and singular_text is not False:
            return [singular_text.upper()], rules
        return [text.upper()], rules

    def _resolve_sound_icon_conflict(self, node_id, node_1_attrs, node_2_attrs):
        graph_1, graph_2 = RebusGraph(), RebusGraph()
        if node_id == 1:
            node_1_attrs_sound = {rule: value for rule, value in node_1_attrs.copy().items() if rule != "icon"}
            node_1_attrs_sound["text"] = list(node_1_attrs_sound["sound"].values())[0].upper()
            node_1_attrs_icon = {rule: value for rule, value in node_1_attrs.copy().items() if rule != "sound"}
            node_1_attrs_icon["text"] = list(node_1_attrs_icon["icon"].values())[0]
            graph_1.add_node(1, **node_1_attrs_sound)
            graph_1.add_node(2, **node_2_attrs)
            graph_2.add_node(1, **node_1_attrs_icon)
            graph_2.add_node(2, **node_2_attrs)
        else:
            node_2_attrs_sound = {rule: value for rule, value in node_2_attrs.copy().items() if rule != "icon"}
            node_2_attrs_sound["text"] = list(node_2_attrs_sound["sound"].values())[0].upper()
            node_2_attrs_icon = {rule: value for rule, value in node_2_attrs.copy().items() if rule != "sound"}
            node_2_attrs_icon["text"] = list(node_2_attrs_icon["icon"].values())[0]
            graph_1.add_node(1, **node_1_attrs)
            graph_1.add_node(2, **node_2_attrs_sound)
            graph_2.add_node(1, **node_1_attrs)
            graph_2.add_node(2, **node_2_attrs_icon)

        graph_2.add_edge(1, 2, rule="NEXT-TO")
        return [graph_1, graph_2]
