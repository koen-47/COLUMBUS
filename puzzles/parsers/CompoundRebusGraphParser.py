import json
import os
import ast

import inflect

from puzzles.patterns.Rule import Rule
from puzzles.RebusGraph import RebusGraph
from util import remove_duplicate_graphs

inflect = inflect.engine()


class CompoundRebusGraphParser:
    """
    Class to parse over a compound word (or a pair of words) to generate a single node graph.
    """
    def __init__(self):
        # Load homophones
        with open(f"{os.path.dirname(__file__)}/../../data/misc/homophones_v2.json", "r") as file:
            self._homophones = json.load(file)
        # Load icons
        with open(f"{os.path.dirname(__file__)}/../../data/misc/icons_v2.json", "r") as file:
            self._icons = {ast.literal_eval(labels): icon for labels, icon in json.load(file).items()}

    def parse(self, c1, c2, is_plural):
        """
        Generates all possible single-node graphs from the specified input words (c1, c2).

        :param c1: first word that will be checked to see if it triggers any rule keywords.
        :param c2: second word that will be checked to see if it triggers any rule keywords.
        :param is_plural: flag to denote if combined word (i.e., f"{c1}{c2}") is plural. This is used to trigger the
        repetition rules. NOTE: this only applies to compound words, not for pairs of words.
        :return: a list of single-node graphs corresponding to each of the generated graphs.
        """
        # Check for patterns for either constituent word
        rules_c1, conflicts_c1 = Rule.find_all(c1, is_plural)
        rules_c2, conflicts_c2 = Rule.find_all(c2, is_plural)

        # Format patterns such that all mutually exclusive rules are handled individually
        if len(conflicts_c1) > 1:
            rules_c1 = [{key: value for key, value in rules_c1.items()
                            if key not in list([conflict_.split("_")[0] for conflict_
                                                in set(conflicts_c1) - {conflict}])}
                           for conflict in conflicts_c1]
        if len(conflicts_c2) > 1:
            rules_c2 = [{key: value for key, value in rules_c2.items()
                            if key not in list([conflict_.split("_")[0] for conflict_
                                                in set(conflicts_c2) - {conflict}])}
                           for conflict in conflicts_c2]

        # List to hold all possible generated puzzles
        graphs = []

        # Generate puzzles by combining both words into one
        for rule_c1 in [rules_c1] if type(rules_c1) is not list else rules_c1:
            graphs += self._generate_rebus(c2, rule_c1, is_plural)
        for rule_c2 in [rules_c2] if type(rules_c2) is not list else rules_c2:
            graphs += self._generate_rebus(c1, rule_c2, is_plural)

        # Generate puzzles by placing both words next to each other
        graphs += self._generate_rebus(c1, {}, is_plural, c2)

        # Remove duplicate puzzles
        graphs = remove_duplicate_graphs(graphs)

        for graph in graphs:
            graph.graph["answer"] = f"{c1}{c2}"

        return graphs

    def _generate_rebus(self, word_1, rules, is_plural, word_2=None):
        """
        Generates a RebusGraph object for the specified word and rules.

        :param word_1: word associated with the node/element that will be in the rebus graph.
        :param rules: rules associated with the element.
        :param is_plural: flag to denote if word_1 is plural.
        :param word_2: second word for a puzzle that places word_1 and word_2 next to each other.
        :return: a generated RebusGraph object.
        """
        # Create RebusGraph object to store the rebus puzzle representation
        graph = RebusGraph()

        # Only generate the graph if there is more than 1 rule (repeat rule is always there)
        if len(rules) > 1:
            text, rules = self._parse_text(word_1, rules, is_plural)

            # If the word triggers both a sound and icon rule, they will be split and generated as two
            # graphs separately.
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
            # Check for homophones
            word_1_homophone = self.parse_homophones(word_1)
            word_2_homophone = self.parse_homophones(word_2)

            # Check for icons
            word_1_icon = self.parse_icon(word_1)
            word_2_icon = self.parse_icon(word_2)

            node_1_attrs = {"text": word_1.upper(), "repeat": 1}
            node_2_attrs = {"text": word_2.upper(), "repeat": 1}

            # Add sound attributes to the nodes
            if word_1_homophone != word_1 or word_2_homophone != word_2:
                if word_1_homophone != word_1:
                    node_1_attrs["sound"] = {word_1: word_1_homophone}
                    node_1_attrs["text"] = word_1_homophone.upper()
                if word_2_homophone != word_2:
                    node_2_attrs["sound"] = {word_2: word_2_homophone}
                    node_2_attrs["text"] = word_2_homophone.upper()

            # Add icon attributes to the nodes.
            if word_1_icon != word_1 or word_2_icon != word_2:
                if word_1_icon != word_1:
                    node_1_attrs["icon"] = {word_1: word_1_icon}
                    node_1_attrs["text"] = word_1_icon.upper()
                if word_2_icon != word_2:
                    node_2_attrs["icon"] = {word_2: word_2_icon}
                    node_2_attrs["text"] = word_2_icon.upper()

            # Return empty list if none of the sound or icon rules are triggered for either of the words.
            if (node_1_attrs == {"text": word_1.upper(), "repeat": 1} and
                    node_2_attrs == {"text": word_2.upper(), "repeat": 1}):
                return []

            # Resolve conflicts for sound and icon attributes in the same node.
            if "sound" in node_1_attrs and "icon" in node_1_attrs:
                return self._resolve_sound_icon_conflict(1, node_1_attrs, node_2_attrs)
            if "sound" in node_2_attrs and "icon" in node_2_attrs:
                return self._resolve_sound_icon_conflict(2, node_1_attrs, node_2_attrs)

            # Add nodes to the graph with the specified attributes and connect them with a NEXT-TO relational rule/edge.
            graph.add_node(1, **node_1_attrs)
            graph.add_node(2, **node_2_attrs)
            graph.add_edge(1, 2, rule="NEXT-TO")
            return [graph]

        # Return empty list if there is no more than 1 rule
        return []

    def parse_homophones(self, text):
        """
        Checks if the specified text is also a homophone.

        :param text: text to match against a homophone.
        :return: the matched homophone (if found), otherwise just the text itself.
        """

        # Replace text with alternative that is phonetically identical
        if text in self._homophones:
            text = self._homophones[text][0]
            return text
        return text

    def parse_icon(self, text):
        """
        Checks if the specified text is also an icon.

        :param text: text to match against an icon.
        :return: the matched icon (if found), otherwise just the text itself.
        """

        # Replace text with alternative that is an icon
        for label, icon in self._icons.items():
            if text.lower() in label:
                text = icon
                return text
        return text

    def _parse_text(self, text, rules, is_plural):
        """
        Checks if the specified text is an icon or a homophone, and adjusts the rules and plurality accordingly.
        Uses the parse_homophone and parse_icon helper functions.

        :param text: text to match against an icon or homophone.
        :param rules: rules to adjust in case icon or homophone is matched.
        :param is_plural: flag to denote if the specified text is plural.
        :return: returns a list in the following format:
        ([output of parse_homophone, output of parse_icon], adjusted rules)
        """
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
        """
        Resolves sound/icon conflicts (if there are any) by creating two separate graphs with sound and icon rules
        separately. Sound and icon rules are not able to co-exist for a single node, because both of these rules
        involve substituting the text associated with this node.

        :param node_id: node ID to distinguish between which of the two nodes in the parse() function contains
        the conflict.
        :param node_1_attrs: attributes of the first node (dictionary).
        :param node_2_attrs: attributes of the second node (dictionary).
        :return: two graphs, where the sound-icon conflict has been split between them (one graph contains the sound
        rule, while the other contains the icon rule).
        """

        # Create two RebusGraph objects to hold both split the conflict into.
        graph_1, graph_2 = RebusGraph(), RebusGraph()

        # Resolve conflicts if the conflicted node is the first one in the parse() function
        if node_id == 1:
            node_1_attrs_sound = {rule: value for rule, value in node_1_attrs.copy().items() if rule != "icon"}
            node_1_attrs_sound["text"] = list(node_1_attrs_sound["sound"].values())[0].upper()
            node_1_attrs_icon = {rule: value for rule, value in node_1_attrs.copy().items() if rule != "sound"}
            node_1_attrs_icon["text"] = list(node_1_attrs_icon["icon"].values())[0]
            graph_1.add_node(1, **node_1_attrs_sound)
            graph_1.add_node(2, **node_2_attrs)
            graph_2.add_node(1, **node_1_attrs_icon)
            graph_2.add_node(2, **node_2_attrs)

        # Resolve the conflicts if the conflicted node is the second one in the parse() function
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
