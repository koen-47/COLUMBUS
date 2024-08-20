import itertools
import json
import os

from puzzles.RebusGraph import RebusGraph
from ..patterns.Rule import Rule
from .CompoundRebusGraphParser import CompoundRebusGraphParser


class PhraseRebusGraphParser:
    """
    Class to parse phrases (i.e., strings with multiple words)
    """

    def __init__(self):
        with open(f"{os.path.dirname(__file__)}/../../data/misc/ignore_words.json", "r") as file:
            self._ignore_words = json.load(file)

    def _is_valid(self, phrase):
        """
        Checks if the specified phrase is suitable to be converted to a rebus graph.

        :param phrase: phrase to check the validity for.
        :return: true/false depending on if the specified phrase is valid.
        """

        # Empty phrases are not valid
        if phrase == "":
            return False

        relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
        phrase_parts = phrase.split()
        n_rel_keywords = [word for word in phrase_parts if word in relational_keywords]

        # Phrases that contain more than one triggered relational rule keyword are not valid
        if len(n_rel_keywords) > 1:
            return False

        # Phrases that start or end with a relational keyword are not valid
        if phrase_parts[0].lower() in relational_keywords:
            return False
        if phrase_parts[-1].lower() in relational_keywords:
            return False

        return True

    def parse(self, phrase):
        """
        Parses a phrase to its rebus graph representation.

        :param phrase: phrase to convert to a rebus graph.
        :return: list containing all the possible combinations of rebus graphs for the specified phrase.
        """

        answer = phrase

        # Remove ignored words from the phrase
        phrase_words = [word for word in phrase.split() if word not in self._ignore_words]
        phrase = " ".join(phrase_words)

        # Check for validity and return None in case of invalidity.
        if not self._is_valid(phrase):
            return None

        # For each word (pair) in the phrase, generate the possible graphs for that word (pair).
        graphs_per_word = self._get_all_graphs_per_word(phrase)
        all_graphs = self._get_all_combinations(graphs_per_word)

        for graph in all_graphs:
            graph.graph["answer"] = answer

        return all_graphs

    def _get_all_combinations(self, graphs_per_words):
        """
        Computes the Cartesian product sequentially between the list of lists with rebus graphs from the
        get_all_graphs_per_word function. For each combination of rebus graphs, these are connected with the relational
        rule keyword found in the original phrase or with a NEXT-TO relational rule.

        :param graphs_per_words: list of lists containing the possible rebus graphs for each word (pair) in the original
        phrase.
        :return: a list of graphs, one for each possible interpretation of the original phrase.
        """
        combinations = list(itertools.product(*graphs_per_words))

        all_graphs = []

        # Iterate over each combination
        for c in combinations:
            relational_nodes = []

            # Start with the first node in this combination
            graph = list(c)[0].copy()
            n_nodes = len(graph.nodes)

            # Iterate over each rebus graph in this combination and start sequentially adding nodes
            for sub_graph in list(c)[1:]:
                # Add the node in this graph to the running graph and connect it with the previously added node with a
                # NEXT-TO relation. We also keep track of where the original relational rule is that was found during
                # the subphrase splitting phase, and add this later as an edge.
                if isinstance(sub_graph, RebusGraph):
                    n_nodes += len(sub_graph.nodes)
                    for node in sub_graph.nodes(data=True):
                        graph.add_node(len(graph.nodes) + 1, **node[1])
                        graph.add_edge(len(graph.nodes) - 1, len(graph.nodes), rule="NEXT-TO")
                else:
                    relational_nodes.append((sub_graph, n_nodes, n_nodes + 1))
            all_graphs.append(graph)

            # Make sure to add an edge with the original relational rule that we found while splitting.
            for edge in relational_nodes:
                graph[edge[1]][edge[2]]["rule"] = edge[0]

        return all_graphs

    def _get_all_graphs_per_word(self, phrase):
        """
        Generate all possible graphs for each word pair in the specified phrase (see the last paragraph of the
        Graph Generation Algorithm subsection in Section 3.2).

        :param phrase: the phrase that will be converted to the possible rebus graphs.
        :return: list of lists, where each inner list contains the possible generated rebus graphs. The outer list is
        indexed by the word pair used to generate the rebus graphs in the inner list.
        """

        # The compound parser used to convert each word pair to a single node graph
        compound_parser = CompoundRebusGraphParser()

        # Subphrase splitting
        divided_words = self._divide_text(phrase)

        graphs_per_word = []
        skip = False
        for words in divided_words:
            # Skip if the word in the split phrase is a relational rule keyword
            for rule, keywords in Rule.get_all_rules()["relational"].items():
                if words[0] in keywords:
                    graphs_per_word.append([rule.upper()])
                    skip = True
                    break
            if skip:
                skip = False
                continue

            # Generate the graph for if the subphrase is only one word. In this case, it is not a word pair and can only
            # trigger the sound/icon rules.
            if len(words) == 1:
                graph = RebusGraph()
                homophone = compound_parser.parse_homophones(words[0])
                icon = compound_parser.parse_icon(words[0])
                node_attrs = {"text": words[0].upper(), "repeat": 1}
                if words[0] != homophone:
                    node_attrs["text"] = homophone.upper()
                    node_attrs["sound"] = {words[0]: homophone}
                if words[0] != icon:
                    node_attrs["icon"] = {words[0]: icon}

                # Resolve sound-icon conflict
                if "sound" in node_attrs and "icon" in node_attrs:
                    graph_1, graph_2 = RebusGraph(), RebusGraph()
                    node_attrs_1 = {rule: value for rule, value in node_attrs.copy().items() if rule != "icon"}
                    node_attrs_2 = {rule: value for rule, value in node_attrs.copy().items() if rule != "sound"}
                    graph_1.add_node(1, **node_attrs_1)
                    graph_2.add_node(1, **node_attrs_2)
                    graphs_per_word.append([graph_1, graph_2])
                    continue
                graph.add_node(1, **node_attrs)
                graphs_per_word.append([graph])
                continue

            # Iterate over the words in the subphrase, converting each word pair to a single node graph returned
            # by the compound graph parser.
            i = 0
            while i < len(words) - 1:
                graphs = compound_parser.parse(c1=words[i], c2=words[i + 1], is_plural=False)
                if len(graphs) > 0:
                    graphs_per_word.append(graphs)
                    words.pop(i)
                    words.pop(i)
                    i -= 1
                else:
                    graph = RebusGraph()
                    graph.add_node(1, text=words[i].upper(), repeat=1)
                    graphs_per_word.append([graph])
                i += 1

            # After converting all word pairs, convert the remaining word of the phrase to a node (since this last part
            # is not a word pair, this can only be converted to a node with a sound/icon rule)
            if i < len(words):
                graph = RebusGraph()
                homophone = compound_parser.parse_homophones(words[i])
                icon = compound_parser.parse_icon(words[i])
                node_attrs = {"text": words[i].upper(), "repeat": 1}
                if words[i] != homophone:
                    node_attrs["text"] = homophone.upper()
                    node_attrs["sound"] = {words[i]: homophone}
                if words[i] != icon:
                    node_attrs["icon"] = {words[i]: icon}

                # Resolve sound-icon conflict
                if "sound" in node_attrs and "icon" in node_attrs:
                    graph_1, graph_2 = RebusGraph(), RebusGraph()
                    node_attrs_1 = {rule: value for rule, value in node_attrs.copy().items() if rule != "icon"}
                    node_attrs_2 = {rule: value for rule, value in node_attrs.copy().items() if rule != "sound"}
                    graph_1.add_node(1, **node_attrs_1)
                    graph_2.add_node(1, **node_attrs_2)
                    graphs_per_word.append([graph_1, graph_2])
                else:
                    graph.add_node(1, **node_attrs)
                    graphs_per_word.append([graph])
        return graphs_per_word

    def _divide_text(self, phrase):
        """
        Divide the text into subphrases, based on the relational keywords in the specified phrase.

        :param phrase: phrase to split into two subphrases.
        :return: list containing the subphrases. The output is of the form:
        [subphrase_1, relational_keyword_1, subphrase_2, relational_keyword_2, ..., relational_keyword_n, subphrase_n+1]
        In COLUMBUS, we only consider an input if it has exactly one relational keyword in the middle of the phrase, so
        the output will always be: [subphrase_1, relational_keyword, subphrase_2].
        """

        # Get all the triggerable relational rule keywords in a single flattened list
        relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]

        # Split the phrase by whitespace
        phrase_words = phrase.split()
        divided_words = []

        # Iterate over each word in the phrase and keep splitting them based on if the current word is a relational
        # rule keyword or not.
        i = 0
        while i < len(phrase_words):
            word = phrase_words[i]
            if word in relational_keywords:
                divided_words.append(phrase_words[:i])
                divided_words.append([phrase_words[i]])
                phrase_words = phrase_words[i + 1:]
                i = -1
            i += 1

        divided_words += [phrase_words]
        return divided_words
