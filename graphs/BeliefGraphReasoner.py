import math
import os
import itertools

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


class BeliefGraphReasoner:
    """
    Class that fixes all the conflicts in a belief graph by converting it to a MaxSAT problem and using a SAT solver
    to compute the optimal amount of truth values to flip.
    """
    def __init__(self, hyperparameters):
        self._hyperparameters = hyperparameters

    def fix_graph(self, graph, verbose=False):
        """
        Fixes the logical inconsistencies in a belief graph. These logical inconsistencies are either XOR or implication
        conflicts (when the given truth assignments result in false for XOR and implication operators).

        :param graph: belief graph (BeliefGraph object) to fix the logical conflicts for.
        :param verbose: flag to denote if intermediate outputs will be printed.
        :return: belief graph (BeliefGraph object) with fixed logical conflicts.
        """
        cnf = self._generate_cnf(graph)
        wcnf = WCNF()
        for clause, weight in cnf:
            if verbose:
                print(f"Clause: {clause} (weight: {weight:.2f})")
            wcnf.append(clause, weight=weight)

        rc2 = RC2(wcnf)
        model = rc2.compute()
        cost = rc2.cost

        if verbose:
            print("Model:", model)
            print(f"Cost:", cost)

        for variable in model:
            node = int(math.fabs(variable))
            node_attrs = graph.nodes[node]
            instantiation = True if variable > 0 else False
            if node_attrs["type"] == "statement":
                node_attrs["value"] = str(instantiation)

        return graph, cost

    def _generate_cnf(self, graph):
        """
        Converts a belief graph to its MaxSAT equivalent.

        :param graph: belief graph (BeliefGraph object) to convert to a MaxSAT problem.
        :return: list of clauses in the MaxSAT problem.
        """
        unit_clauses = []
        for statement_node in graph.nodes:
            node_attrs = graph.nodes[statement_node]
            if node_attrs["type"] == "statement":
                value = node_attrs["value"] == "True"
                confidence = float(node_attrs["confidence"])

                unit_clause = statement_node if value else -statement_node
                unit_clauses.append(
                    ([unit_clause], 1.)
                )

                neg_unit_clause = -unit_clause
                unit_clauses.append(
                    ([neg_unit_clause], math.exp(-confidence))
                )

        xor_clauses = []
        for rule_node in graph.nodes:
            node_attrs = graph.nodes[rule_node]
            if node_attrs["type"] == "rule" and node_attrs["is_xor"]:
                node, neg_node = node_attrs["connected_nodes"]
                clause_1 = ([node, neg_node], node_attrs["confidence"])
                clause_2 = ([-node, -neg_node], node_attrs["confidence"])
                xor_clauses.append(clause_1)
                xor_clauses.append(clause_2)

        rule_clauses = []
        for rule_node in graph.nodes:
            node_attrs = graph.nodes[rule_node]
            if node_attrs["type"] == "rule" and not node_attrs["is_xor"]:
                premise_nodes = {premise: graph.nodes[premise]["value"] == "True" for premise in
                                 node_attrs["connected_nodes"]["premises"]}
                hypothesis_node = {node_attrs["connected_nodes"]["hypothesis"][0]:
                                   graph.nodes[node_attrs["connected_nodes"]["hypothesis"][0]]["value"] == "True"}
                clauses = list([-variable for variable in premise_nodes.keys()]) + list(hypothesis_node.keys())
                is_satisfied = not all(premise_nodes.values()) or list(hypothesis_node.values())[0]
                weight = 1. if is_satisfied else math.exp(-node_attrs["confidence"])
                rule_clauses.append((clauses, weight))

        orig_hypotheses = graph.get_original_hypotheses()
        hard_constraint = (orig_hypotheses, math.inf)
        mc_clauses = [hard_constraint]

        if len(orig_hypotheses) > 1:
            for orig_hypotheses_pair in itertools.combinations(orig_hypotheses, 2):
                soft_constraint = ([-orig_hypotheses_pair[0], -orig_hypotheses_pair[1]], self._hyperparameters["c_mc"])
                mc_clauses.append(soft_constraint)

        all_clauses = unit_clauses + xor_clauses + rule_clauses + mc_clauses
        return all_clauses
