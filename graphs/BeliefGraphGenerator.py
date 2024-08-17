import json
import math
import itertools
import os
import inflect

from graphs.GPTPrompter import GPTPrompter
from graphs.GeminiPrompter import GeminiPrompter
from graphs.BeliefGraph import BeliefGraph

p = inflect.engine()

class BeliefGraphGenerator:
    def __init__(self, puzzle, n_examples, hypotheses, hyperparameters, max_depth=3, model="gpt-4o-mini"):
        self._images = [puzzle]
        self._n_examples = n_examples
        self._hypotheses = hypotheses
        self._hypothesis_template = "The word/phrase conveyed in this image is \"{}\"."
        self._max_depth = max_depth
        self._hyperparameters = hyperparameters

        if model == "gpt-4o" or model == "gpt-4o-mini":
            self._prompter = GPTPrompter(puzzle, n_examples, model=model)
        elif model == "gemini-1.5-flash" or model == "gemini-1.5-pro":
            self._prompter = GeminiPrompter(puzzle, n_examples, model=model)

    def generate_graph(self, verbose=False):
        graph = BeliefGraph()
        for hypothesis in self._hypotheses:
            hypothesis = self._hypothesis_template.format(hypothesis)
            if verbose:
                print(f"Generating graph for hypothesis: \"{hypothesis}\"")
            self._extend_graph(hypothesis, 0, self._max_depth, graph)
        graph.connect_rules_and_statements()
        return graph

    def _extend_graph(self, statement, depth, max_depth, graph):
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
        premises = self._prompter.generate_premise_from_hypothesis(hypothesis)
        premises = [" ".join(premise.split(" ")[1:]).strip() for premise in premises.split("\n")]
        premises = [premise for premise in premises if premise != ""]
        print(f"Premises:\n{premises}")
        return premises

    def _score_statement(self, statement):
        value, prob = self._prompter.score_statement(statement)
        prob = math.exp(self._hyperparameters["k"] * (prob - 1))
        print(f"Score statement:", value, prob)
        return value, prob

    def _generate_negated_statement(self, statement):
        negated = self._prompter.generate_negated_statement(statement)
        print("Negated:", negated)
        return negated

    def _score_rule(self, premises, hypothesis, is_xor=False, is_mc=False):
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
        print("Score rule:", prob)
        return prob
