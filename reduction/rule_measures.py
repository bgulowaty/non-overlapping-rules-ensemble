from typing import Set

import numpy as np
from attr import attrs, attrib

from .api import Rule, Statement, Relation
from .classification import to_instance

EPS = np.finfo(float).eps

def covered_by_statements(rule, x):
    return {idx for idx, x_single in enumerate(x) if rule.describes(to_instance(x_single))}


def same_label(rule, y):
    return {idx for idx, y_single in enumerate(y) if rule.distribution_or_class == y_single}


@attrs()
class BayesianRuleMeasures:
    covered_sample_indicies: Set[int] = attrib()
    same_lable_indicies: Set[int] = attrib()
    all_indicies : Set[int] = attrib()

    @classmethod
    def create(cls, rule, x, y):
        covered_sample_indicies = covered_by_statements(rule, x)
        same_label_indicies = same_label(rule, y)
        all_indicies = set(range(len(x)))

        return cls(covered_sample_indicies, same_label_indicies, all_indicies)

    def a(self) -> float:
        return len(self.same_lable_indicies.intersection(self.covered_sample_indicies))

    def b(self) -> float:
        return len(self.same_lable_indicies.difference(self.covered_sample_indicies))

    def c(self) -> float:
        return len(self.covered_sample_indicies.difference(self.same_lable_indicies))

    def d(self) -> float:
        return len(self.all_indicies.difference(self.same_lable_indicies).difference(self.covered_sample_indicies))

    def s_measure(self) -> float:
        return (self.a() + EPS) / (self.a() + self.c() + EPS) - (self.b() + EPS) / (self.b() + self.d() + EPS)

    def n_measure(self) -> float:
        return (self.a() + EPS) / (self.a() + self.c() + EPS) - (self.c() + EPS) / (self.c() + self.d() + EPS)


def test_measures_1():
    rule = Rule(statements=[Statement(0, Relation.LEQ, threshold=5)], distribution_or_class=0)

    x = [[0], [1], [2], [6], [7], [8]]
    y = [0, 0, 1, 0, 1, 0]

    measures = BayesianRuleMeasures.create(rule, x, y)

    assert measures.a() == 2
    assert measures.b() == 2
    assert measures.c() == 1
    assert measures.d() == 1

