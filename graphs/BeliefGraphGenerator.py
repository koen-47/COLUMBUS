import json
import math
import itertools
import os
import inflect

from graphs.Prompter import Prompter
from graphs.BeliefGraph import BeliefGraph

p = inflect.engine()

class BeliefGraphGenerator:
    def __init__(self, puzzle, n_examples, hypotheses, hyperparameters, max_depth=3, model="gpt-4o-mini"):
        self._images = [puzzle]
        self._n_examples = n_examples
        self._hypotheses = hypotheses
        self._hypothesis_template = "The word/phrase conveyed in this image is \"{}\"."
        self._prompter = Prompter(model=model)
        self._max_depth = max_depth
        self._hyperparameters = hyperparameters

        with open(f"{os.path.dirname(__file__)}/prompts.json", "r") as file:
            self._prompts = json.load(file)

        if self._n_examples > 0:
            examples = self._prompts["few_shot_examples"][str(self._n_examples)]
            examples_desc = "".join(f"\n- Image {i+1}: {example["description"]}" for i, example in enumerate(examples))
            preprompt_template = self._prompts["prompt_templates"]["few_shot"]["preprompt"]
            if self._n_examples == 1:
                preprompt_template = preprompt_template["1"]
                self._preprompt = preprompt_template.format(*[examples_desc])
            else:
                preprompt_template = preprompt_template["2+"]
                cardinal_num = p.number_to_words(self._n_examples)
                self._preprompt = preprompt_template.format(*[cardinal_num, examples_desc])
            examples_images = [f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/{example["image"]}"
                               for example in examples]
            self._images = examples_images + [puzzle]
        else:
            self._preprompt = self._prompts["prompt_templates"]["zero_shot"]["preprompt"]

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
        text = ""
        if self._n_examples == 0:
            prompt_template = self._prompts["prompt_templates"]["zero_shot"]["generate_premises"]
            text = prompt_template.format(*[self._preprompt, hypothesis])
        elif self._n_examples > 0:
            prompt_template = self._prompts["prompt_templates"]["few_shot"]["generate_premises"]
            ordinal_num = p.number_to_words(p.ordinal(len(self._images)))
            text = prompt_template.format(*[self._preprompt, ordinal_num, len(self._images), hypothesis, len(self._images)])

        # print("Generate premise:", text)
        response = self._prompter.send_prompt(text, image_paths=self._images)
        premises = response["choices"][0]["message"]["content"]
        premises = [" ".join(premise.split(" ")[1:]) for premise in premises.split("\n")]
        return premises

    def _score_statement(self, statement):
        text = ""
        if self._n_examples == 0:
            prompt_template = self._prompts["prompt_templates"]["zero_shot"]["score_statement"]
            text = prompt_template.format(*[self._preprompt, statement])
        elif self._n_examples > 0:
            prompt_template = self._prompts["prompt_templates"]["few_shot"]["score_statement"]
            ordinal_num = p.number_to_words(p.ordinal(len(self._images)))
            text = prompt_template.format(*[self._preprompt, ordinal_num, len(self._images), len(self._images), statement])

        # print("Score statement:", text)
        response = self._prompter.send_prompt(text, image_paths=self._images, max_tokens=1)
        content = response["choices"][0]["logprobs"]["content"][0]
        prob = 10 ** float(content["logprob"])
        prob = math.exp(self._hyperparameters["k"] * (prob - 1))
        value = content["token"]
        return value, prob

    def _generate_negated_statement(self, statement):
        prompt_template = self._prompts["prompt_templates"]["generate_negation"]
        text = prompt_template.format(*[statement])
        # print("Negation:", text)
        response = self._prompter.send_prompt(text)
        negated = response["choices"][0]["message"]["content"]
        return negated

    def _score_rule(self, premises, hypothesis, is_xor=False, is_mc=False):
        premises_text = "".join([f"\n- {premise}" for premise in premises if premise != ""])
        text = ""
        if self._n_examples == 0:
            prompt_template = self._prompts["prompt_templates"]["zero_shot"]["score_rule"]
            text = prompt_template.format(*[self._preprompt, premises_text, hypothesis])
        elif self._n_examples > 0:
            prompt_template = self._prompts["prompt_templates"]["few_shot"]["score_rule"]
            ordinal_num = p.number_to_words(p.ordinal(len(self._images)))
            text = prompt_template.format(*[self._preprompt, ordinal_num, len(self._images), len(self._images),
                                            premises_text, hypothesis])

        # print("Score rule:", text)
        response = self._prompter.send_prompt(text, image_paths=self._images, max_tokens=1)
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
