import json
import math
import os
import re

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from util import get_node_attributes


class BeliefGraph(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    def get_original_hypotheses(self):
        orig_hypothesis_nodes = []
        for node in self.nodes:
            node_attrs = self.nodes[node]
            if node_attrs["type"] == "statement" and node_attrs["is_orig"]:
                orig_hypothesis_nodes.append(node)
        return orig_hypothesis_nodes

    def add_statement_node(self, statement, value, confidence, is_orig=False, return_node=False):
        node_id = self.number_of_nodes() + 1
        attrs = {"type": "statement", "statement": statement, "value": value, "confidence": confidence,
                 "is_orig": is_orig}
        self.add_node(node_id, **attrs)
        if return_node:
            return node_id

    def add_rule_node(self, statements, confidence, is_xor=False, is_mc=False, return_node=False):
        node_id = self.number_of_nodes() + 1
        attrs = {"type": "rule", "confidence": confidence, "is_xor": is_xor, "statements": statements}
        if not is_xor and not is_mc:
            attrs["statements"] = {"premises": statements[:-1], "hypothesis": statements[-1]}
        self.add_node(node_id, **attrs)
        if return_node:
            return node_id

    def connect_rules_and_statements(self):
        for rule_node in self.nodes:
            rule_node_attrs = self.nodes[rule_node]
            if rule_node_attrs["type"] == "rule" and not rule_node_attrs["is_xor"]:
                premises = rule_node_attrs["statements"]["premises"]
                hypothesis = rule_node_attrs["statements"]["hypothesis"]
                connected_nodes = {"premises": [], "hypothesis": []}
                for statement_node in self.nodes:
                    statement_node_attrs = self.nodes[statement_node]
                    if statement_node_attrs["type"] == "statement":
                        for premise in premises:
                            if premise == statement_node_attrs["statement"]:
                                connected_nodes["premises"].append(statement_node)
                                self.add_edge(statement_node, rule_node)
                        if hypothesis == statement_node_attrs["statement"]:
                            connected_nodes["hypothesis"].append(statement_node)
                            self.add_edge(rule_node, statement_node)
                rule_node_attrs["connected_nodes"] = connected_nodes
                del rule_node_attrs["statements"]
            elif rule_node_attrs["type"] == "rule" and rule_node_attrs["is_xor"]:
                connected_nodes = []
                for statement_node in self.nodes:
                    statement_node_attrs = self.nodes[statement_node]
                    if (statement_node_attrs["type"] == "statement" and
                            statement_node_attrs["statement"] in rule_node_attrs["statements"]):
                        connected_nodes.append(statement_node)
                        self.add_edge(rule_node, statement_node)
                rule_node_attrs["connected_nodes"] = connected_nodes
                del rule_node_attrs["statements"]

    def visualize(self, show=False, save_path=None):
        pos = nx.nx_agraph.graphviz_layout(self, prog="dot")
        plt.figure(figsize=(12, 12))

        statement_nodes = [node for node in self.nodes if self.nodes[node]["type"] == "statement"]
        true_statement_nodes = [node for node in statement_nodes if self.nodes[node]["value"] == "True"]
        false_statement_nodes = [node for node in statement_nodes if self.nodes[node]["value"] == "False"]
        rule_nodes = [node for node in self.nodes if self.nodes[node]["type"] == "rule"]

        nx.draw_networkx_nodes(self, pos, nodelist=true_statement_nodes, node_shape="s", node_color="green",
                               node_size=[400] * len(true_statement_nodes))
        nx.draw_networkx_nodes(self, pos, nodelist=false_statement_nodes, node_shape="s", node_color="red",
                               node_size=[400] * len(false_statement_nodes))
        nx.draw_networkx_nodes(self, pos, nodelist=rule_nodes, node_shape="o", node_color="black")

        statement_labels = {node: node for node in statement_nodes}
        nx.draw_networkx_labels(self, pos, statement_labels, font_size=8, font_color="white")
        rule_labels = {node: node for node in rule_nodes}
        nx.draw_networkx_labels(self, pos, rule_labels, font_size=8, font_color="white")

        nx.draw_networkx_edges(self, pos, arrows=True)

        xor_edges = [edge for edge in self.edges if
                     (self.nodes[edge[0]]["type"] == "rule" and self.nodes[edge[0]]["is_xor"] is True) or
                     (self.nodes[edge[1]]["type"] == "rule" and self.nodes[edge[1]]["is_xor"] is True)]
        xor_edges = {edge: "XOR" for edge in xor_edges}
        nx.draw_networkx_edge_labels(self, pos, edge_labels=xor_edges, font_size=8)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
        if show:
            plt.show()

    def get_answer(self):
        orig_hypotheses = self.get_original_hypotheses()
        answers = [(self.nodes[h]["statement"], self.nodes[h]["confidence"])
                   for h in orig_hypotheses if self.nodes[h]["value"] == "True"]
        if len(answers) == 0:
            return None
        best_answer = max(answers, key=lambda x: x[1])[0]
        best_answer = re.findall(r'"([^"]*)"', best_answer)[0]
        return best_answer

    def __str__(self):
        string = "\n=== GRAPH DESCRIPTION ===\n"
        for i, node in enumerate(self.nodes):
            node_attrs = self.nodes[node]
            if node_attrs["type"] != "statement":
                continue
            string += f"STATEMENT (id: {i + 1}, value: {node_attrs["value"]}, confidence: {node_attrs["confidence"]:.2f})"
            string += f" - \"{node_attrs["statement"]}\"\n"

        for i, node in enumerate(self.nodes):
            node_attrs = self.nodes[node]
            if node_attrs["type"] != "rule" or node_attrs["is_xor"]:
                continue
            string += f"\nRULE NODE (id: {i + 1}, confidence: {node_attrs["confidence"]:.2f})"
            premises = [self.nodes[premise]["statement"] for premise in node_attrs["connected_nodes"]["premises"]]
            hypothesis = self.nodes[node_attrs["connected_nodes"]["hypothesis"][0]]["statement"]
            string += "\n\tPremises:\n" + "\n".join([f"\t\t- {statement}" for statement in premises])
            string += f"\n\tHypothesis:\n\t\t- {hypothesis}\n"

        for i, node in enumerate(self.nodes):
            node_attrs = self.nodes[node]
            if node_attrs["type"] != "rule" or not node_attrs["is_xor"]:
                continue
            statements = [self.nodes[statement]["statement"] for statement in node_attrs["connected_nodes"]]
            string += f"\nXOR RULE NODE (id: {i + 1}, confidence: {node_attrs["confidence"]:.2f})"
            string += f"\n\t- {statements[0]}\n\t- {statements[1]}\n"

        return string
