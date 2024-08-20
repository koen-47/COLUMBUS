"""
File to generate a belief graph.
"""

import math
import inflect

from graphs.GPTPrompter import GPTPrompter
from graphs.BeliefGraph import BeliefGraph

p = inflect.engine()


class BeliefGraphGenerator:
    """
    Class to generate a belief graph.
    """
    def __init__(self, puzzle, n_examples, hypotheses, hyperparameters, max_depth=3, model="gpt-4o-mini"):
        self._images = [puzzle]
        self._n_examples = n_examples
        self._hypotheses = hypotheses
        self._hypothesis_template = "The word/phrase conveyed in this image is \"{}\"."
        self._max_depth = max_depth
        self._hyperparameters = hyperparameters

        # We only evaluate belief graphs on GPT-4o (mini) because the Gemini API does not produce token probabilities
        # for its generated responses, which is crucial to evaluate this method fairly (see Table 2 in the paper).
        if model == "gpt-4o" or model == "gpt-4o-mini":
            self._prompter = GPTPrompter(puzzle, n_examples, model=model)

    def generate_graph(self, verbose=False):
        """
        Generates a belief graph (instantiated BeliefGraph object).
        :param verbose: flag to denote if intermediate outputs will the printed.
        :return: generated belief graph.
        """
        graph = BeliefGraph()
        for hypothesis in self._hypotheses:
            hypothesis = self._hypothesis_template.format(hypothesis)
            if verbose:
                print(f"Generating graph for hypothesis: \"{hypothesis}\"")
            self._extend_graph(hypothesis, 0, self._max_depth, graph)
        graph.connect_rules_and_statements()
        return graph

    def _extend_graph(self, statement, depth, max_depth, graph):
        """
        Recursive function to add new nodes to the belief graph.

        :param statement: string of the statement to evaluate and add.
        :param depth: current depth of the belief graph.
        :param max_depth: maximum depth to add new nodes in the belief graph.
        :param graph: BeliefGraph object to add new nodes to.
        :return: base case is to return 0 if depth >= max_depth.
        """
        statement_value, statement_prob = self._score_statement(statement)
        graph.add_statement_node(statement, statement_value, statement_prob, is_orig=depth == 0)

        neg_statement = self._generate_negated_statement(statement)
        neg_value, neg_prob = self._score_statement(neg_statement)

        if math.fabs(statement_prob - neg_prob) > self._hyperparameters["m_xor"] and depth < max_depth:
            graph.add_statement_node(neg_statement, neg_value, neg_prob)
            statements = [statement, neg_statement]
            graph.add_rule_node(statements, confidence=self._hyperparameters["c_xor"], is_xor=True)

        if depth < max_depth:
            premises = self._generate_premise_from_hypothesis(statement)
            rule_prob = self._score_rule(premises, statement)
            graph.add_rule_node(premises + [statement], confidence=rule_prob)
            for premise in premises:
                self._extend_graph(premise, depth + 1, max_depth, graph)

            if math.fabs(statement_prob - neg_prob) > self._hyperparameters["m_xor"]:
                premises = self._generate_premise_from_hypothesis(neg_statement)
                rule_prob = self._score_rule(premises, neg_statement)
                graph.add_rule_node(premises + [neg_statement], confidence=rule_prob)
                for premise in premises:
                    self._extend_graph(premise, depth + 1, max_depth, graph)
        else:
            return 0

    def _generate_premise_from_hypothesis(self, hypothesis):
        """
        Call GPT-4o API to generate premises from a hypothesis.
        :param hypothesis: string of hypothesis to generate premises for.
        :return: a list of generated premises (strings).
        """
        premises = self._prompter.generate_premise_from_hypothesis(hypothesis)
        premises = [" ".join(premise.split(" ")[1:]).strip() for premise in premises.split("\n")]
        premises = [premise for premise in premises if premise != ""]
        return premises

    def _score_statement(self, statement):
        """
        Scores a statement with the assigned truth value and confidence in this truth value.
        :param statement: string of statement to score.
        :return: assigned truth value and confidence in this truth value.
        """
        value, prob = self._prompter.score_statement(statement)
        prob = math.exp(self._hyperparameters["k"] * (prob - 1))
        return value, prob

    def _generate_negated_statement(self, statement):
        """
        Generate a negated statement.
        :param statement: string of statement to negate.
        :return: a negated statement of the specified statement.
        """
        negated = self._prompter.generate_negated_statement(statement)
        return negated

    def _score_rule(self, premises, hypothesis, is_xor=False, is_mc=False):
        """
        Scores a rule by computing a confidence that the premises imply the hypothesis.

        :param premises: list of premises (strings).
        :param hypothesis: hypothesis (string).
        :param is_xor: flag to denote if the rule node to score is an XOR rule node.
        :param is_mc: flag to denote if the rule to score is a multiple-choice constraint node.
        :return: confidence (probability) that the premises imply the hypothesis.
        """
        prob = self._prompter.score_rule(premises, hypothesis)

        k_type = self._hyperparameters["k_entailer"]
        if is_xor:
            k_type = self._hyperparameters["k_xor"]
        elif is_mc:
            k_type = self._hyperparameters["k_mc"]

        t_type = self._hyperparameters["t_entailer"]
        if is_xor:
            t_type = self._hyperparameters["t_xor"]
        elif is_mc:
            t_type = self._hyperparameters["t_mc"]

        prob = t_type * math.exp(k_type * (prob - 1))
        return prob
