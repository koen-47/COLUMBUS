import math
import itertools

from graphs.Prompter import Prompter
from graphs.BeliefGraph import BeliefGraph


class BeliefGraphGenerator:
    def __init__(self, image_path, hypotheses, hyperparameters, max_depth=3, model="gpt-4o-mini"):
        self._image_path = image_path
        self._hypotheses = hypotheses
        self._hypothesis_template = "The word/phrase conveyed in this image is \"{}\"."
        self._prompter = Prompter(model=model)
        self._max_depth = max_depth
        self._hyperparameters = hyperparameters

    def generate_graph(self, verbose=False):
        graph = BeliefGraph()
        for hypothesis in self._hypotheses:
            hypothesis = self._hypothesis_template.format(hypothesis)
            if verbose:
                print(f"Generating graph for hypothesis: \"{hypothesis}\"")
            self._extend_graph(hypothesis, 0, self._max_depth, graph)
        graph.connect_rules_and_statements()
        # graph.remove_xor_leaf_nodes()
        return graph

    # def _extend_graph(self, statement, depth, max_depth, graph):
    #     if depth > max_depth:
    #         return 0
    #
    #     statement_value, statement_prob = self._score_statement(statement)
    #     graph.add_statement_node(statement, statement_value, statement_prob, is_orig=depth == 0)
    #
    #     neg_statement = self._generate_negated_statement(statement)
    #     neg_value, neg_prob = self._score_statement(neg_statement)
    #
    #     if math.fabs(statement_prob - neg_prob) > self._hyperparameters["m_xor"]:
    #         graph.add_statement_node(neg_statement, neg_value, neg_prob)
    #         statements = [statement, neg_statement]
    #         graph.add_rule_node(statements, confidence=self._hyperparameters["c_xor"], is_xor=True)
    #         graph.add_edge(statement, neg_statement)
    #
    #     if depth < max_depth:
    #         premises = self._generate_premise_from_hypothesis(statement)
    #         rule_prob = self._score_rule(premises, statement)
    #         rule_node = graph.add_rule_node(premises + [statement], confidence=rule_prob, return_node=True)
    #         graph.add_edge(rule_node, statement)
    #         for premise in premises:
    #             self._extend_graph(premise, depth + 1, max_depth, graph)
    #             graph.add_edge(premise, rule_node)

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
        text = (f"You are given an image of a rebus puzzle. It consists of text or icons that is used to convey a "
                f"word or phrase. It needs to be solved through creative thinking."
                f"You are also given the hypothesis: \"{hypothesis}\". "
                f"In relation to the given puzzle, explain this hypothesis with a 2-step reasoning chain. "
                f"Keep your response concise and direct and respond in the following format:\n"
                f"- [premise 1]\n"
                f"- [premise 2]\n"
                f"- etc.")
        response = self._prompter.send_prompt(text, image_path=self._image_path)
        premises = response["choices"][0]["message"]["content"]
        premises = [" ".join(premise.split(" ")[1:]) for premise in premises.split("\n")]
        return premises

    def _score_statement(self, statement):
        text = (f"You are given an image of a rebus puzzle. It consists of text or icons that is used to convey a "
                f"word or phrase. It needs to be solved through creative thinking."
                f"Respond with either true or false. Based on the given puzzle, is the following statement "
                f"true or false: {statement}")
        response = self._prompter.send_prompt(text, image_path=self._image_path, max_tokens=1)
        content = response["choices"][0]["logprobs"]["content"][0]
        prob = 10 ** float(content["logprob"])
        prob = math.exp(self._hyperparameters["k"] * (prob - 1))
        value = content["token"]
        return value, prob

    def _generate_negated_statement(self, statement):
        text = f"Respond concisely and directly. Negate the following statement: {statement}"
        response = self._prompter.send_prompt(text)
        negated = response["choices"][0]["message"]["content"]
        return negated

    def _score_rule(self, premises, hypothesis, is_xor=False, is_mc=False):
        premises_text = "".join([f"\n- {premise}" for premise in premises])
        text = (f"You are given an image of a rebus puzzle. It consists of text or icons that is used to convey "
                f"a word or phrase. It needs to be solved through creative thinking."
                f"You are also given a set of premises and a hypothesis. Based on the given puzzle, do you think "
                f"the following premises accurately explain the following hypothesis? "
                f"Respond with either true or false. "
                f"\n\nPremises: {premises_text}"
                f"\nHypothesis: {hypothesis}")
        print(text)
        response = self._prompter.send_prompt(text, image_path=self._image_path, max_tokens=1)
        content = response["choices"][0]["logprobs"]["content"][0]
        prob = 10 ** float(content["logprob"])
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
