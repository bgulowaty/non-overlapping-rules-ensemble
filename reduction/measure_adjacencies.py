from __future__ import annotations

from itertools import groupby, combinations, product
from typing import Set, Collection, List, Optional, Tuple

import pytest
from attr import attrs
from joblib import Parallel, delayed
from numpy import inf as INFINITY

from .api import Rule, Statement, Relation, RulesAdjacencyMeasurer, AdjacentOrNot


@attrs(auto_attribs=True, frozen=True)
class Span:
    upper_bound: float = None
    lower_bound: float = None

    def overlaps(self, span: Span) -> bool:
        return span.lower_bound <= self.lower_bound <= span.upper_bound \
               or span.lower_bound <= self.upper_bound <= span.upper_bound \
               or self.lower_bound <= span.lower_bound <= self.upper_bound \
               or self.lower_bound <= span.upper_bound <= self.upper_bound


def calculate_span_for_statements(statements: Set[Statement]) -> Set[Span]:
    statements_sorted_by_threshold = sorted(statements, key=lambda s: s.threshold)

    statements_without_consecutive_same_relations: List[Statement] = []
    for statement in statements_sorted_by_threshold:
        if len(statements_without_consecutive_same_relations) == 0:
            statements_without_consecutive_same_relations.append(statement)
        elif statements_without_consecutive_same_relations[-1].relation == statement.relation:
            if statement.relation == Relation.MT:
                pass
            else:
                statements_without_consecutive_same_relations[-1] = statement
        else:
            statements_without_consecutive_same_relations.append(statement)

    spans = []

    buffer: Optional[Statement] = None
    for statement in statements_without_consecutive_same_relations:
        if statement.relation == Relation.LEQ:
            if len(spans) == 0 and buffer is None:
                spans.append(Span(statement.threshold, -INFINITY))
            elif buffer is not None:  # must me MT
                spans.append(Span(statement.threshold, buffer.threshold))
                buffer = None
        if statement.relation == Relation.MT:
            buffer = statement

    if buffer != None:
        spans.append(Span(INFINITY, buffer.threshold))
    return set(spans)


def statements_overlap(statements1: Set[Statement], statements2: Set[Statement]) -> bool:
    spans1 = calculate_span_for_statements(statements1)
    spans2 = calculate_span_for_statements(statements2)

    all_spans_product = product(spans1, spans2)

    return any([span1.overlaps(span2) for span1, span2 in all_spans_product])


class OverlappingMeasurer(RulesAdjacencyMeasurer):

    def measure(self, rule1: Rule, rule2: Rule) -> AdjacentOrNot:
        statements_by_feature_idx_1 = {k: list(v) for k, v in
                                       groupby(sorted(rule1.statements, key=lambda s: s.feature_idx),
                                               lambda statement: statement.feature_idx)}
        statements_by_feature_idx_2 = {k: list(v) for k, v in
                                       groupby(sorted(rule2.statements, key=lambda s: s.feature_idx),
                                               lambda statement: statement.feature_idx)}

        overlaps = all(
            [statements_overlap(stat1, stat2) for feat1, stat1 in statements_by_feature_idx_1.items()
             for feat2, stat2 in statements_by_feature_idx_2.items() if feat1 == feat2])

        return AdjacentOrNot.ADJACENT if overlaps is True else AdjacentOrNot.NOT_ADJACENT


def measure(combination_tuple: Tuple[Rule, Rule]) -> AdjacentOrNot:
    return OverlappingMeasurer().measure(combination_tuple[0], combination_tuple[1])


def measure_rules(all_rules: Collection[Rule], n_jobs: int = -1):
    rule_combinations = list(combinations(all_rules, 2))

    if n_jobs == 1:
        measured_combinations = [
            (combination, OverlappingMeasurer().measure(combination[0], combination[1]))
            for combination in rule_combinations
        ]
    else:
        measured_combinations = Parallel(n_jobs=n_jobs)(
            delayed(lambda comb: (comb, OverlappingMeasurer().measure(comb[0], comb[1])))(combination) for combination in
            rule_combinations)

    return dict(measured_combinations)


def test_span_calculation():
    statements = {
        Statement(0, Relation.LEQ, 1),
        Statement(0, Relation.MT, 2),
        Statement(0, Relation.LEQ, 3),
        Statement(0, Relation.LEQ, 4),
        Statement(0, Relation.MT, 5),
        Statement(0, Relation.LEQ, 6),
    }

    spans = calculate_span_for_statements(statements)

    assert {Span(1, -INFINITY), Span(4, 2), Span(6, 5)} == spans


def test_span_calculation_2():
    statements = {
        Statement(0, Relation.MT, 0),
        Statement(0, Relation.LEQ, 1),
        Statement(0, Relation.LEQ, 2),
        Statement(0, Relation.LEQ, 3),
        Statement(0, Relation.MT, 4),
        Statement(0, Relation.LEQ, 5),
        Statement(0, Relation.LEQ, 6),
        Statement(0, Relation.MT, 7),
    }

    spans = calculate_span_for_statements(statements)

    assert {Span(3, 0), Span(6, 4), Span(INFINITY, 7)} == spans


def test_measurer_adjacent():
    rule1 = Rule(
        [
            Statement(0, Relation.MT, 1),
            Statement(0, Relation.LEQ, 3),

        ],
        1
    )

    rule2 = Rule(
        [
            Statement(1, Relation.MT, 10),
            Statement(1, Relation.LEQ, 20),

        ],
        1
    )

    assert OverlappingMeasurer().measure(rule1, rule2) == AdjacentOrNot.ADJACENT
    assert OverlappingMeasurer().measure(rule2, rule1) == AdjacentOrNot.ADJACENT


def test_measurer_not_adjacent():
    rule1 = Rule(
        [
            Statement(0, Relation.MT, 1),
            Statement(0, Relation.LEQ, 3),
            Statement(1, Relation.LEQ, 3),
        ],
        1
    )

    rule2 = Rule(
        [
            Statement(0, Relation.MT, 10),
            Statement(0, Relation.LEQ, 20),
            Statement(1, Relation.MT, 5),
            Statement(1, Relation.LEQ, 10),

        ],
        1
    )

    assert OverlappingMeasurer().measure(rule1, rule2) == AdjacentOrNot.NOT_ADJACENT
    assert OverlappingMeasurer().measure(rule2, rule1) == AdjacentOrNot.NOT_ADJACENT


def test_measurer_adjacent_2():
    rule1 = Rule(
        [
            Statement(0, Relation.MT, 1),
            Statement(0, Relation.LEQ, 3),
            Statement(2, Relation.MT, 1),
            Statement(2, Relation.LEQ, 3),

        ],
        1
    )

    rule2 = Rule(
        [
            Statement(0, Relation.MT, 1),
            Statement(0, Relation.LEQ, 3),
            Statement(1, Relation.MT, 1),
            Statement(1, Relation.LEQ, 3),

        ],
        1
    )

    assert OverlappingMeasurer().measure(rule1, rule2) == AdjacentOrNot.ADJACENT
    assert OverlappingMeasurer().measure(rule2, rule1) == AdjacentOrNot.ADJACENT


def test_measurer_not_adjacent_real():
    rule1 = Rule([
        Statement(feature_idx=3, relation=Relation.LEQ, threshold=2.5),
        Statement(feature_idx=2, relation=Relation.MT, threshold=1.1),
        Statement(feature_idx=0, relation=Relation.MT, threshold=5.75),
        Statement(feature_idx=1, relation=Relation.LEQ, threshold=3.700000047683716),
        Statement(feature_idx=0, relation=Relation.LEQ, threshold=7.9),
        Statement(feature_idx=2, relation=Relation.LEQ, threshold=4.950000047683716),
        Statement(feature_idx=3, relation=Relation.MT, threshold=1.699999988079071),
        Statement(feature_idx=1, relation=Relation.MT, threshold=2.0)
    ],
        0)
    rule2 = Rule([
        Statement(feature_idx=3, relation=Relation.LEQ, threshold=2.5),
        Statement(feature_idx=2, relation=Relation.LEQ, threshold=6.9),
        Statement(feature_idx=2, relation=Relation.MT, threshold=2.350000023841858),
        Statement(feature_idx=3, relation=Relation.MT, threshold=1.6500000357627869)

    ],
        1)

    assert OverlappingMeasurer().measure(rule1, rule2) == AdjacentOrNot.ADJACENT
    assert OverlappingMeasurer().measure(rule2, rule1) == AdjacentOrNot.ADJACENT


@pytest.mark.parametrize("span1,span2,expected", [
    (Span(5, 3), Span(6, 2), True),
    (Span(5, 3), Span(0, -5), False),
    (Span(10, 0), Span(15, 5), True),
    (Span(10, 0), Span(5, -10), True),
    (Span(10, 0), Span(20, 15), False),
    (Span(10, 0), Span(-10, -20), False),

])
def test_span_overlapping(span1: Span, span2: Span, expected: bool):
    assert span1.overlaps(span2) == expected
    assert span2.overlaps(span1) == expected


@pytest.mark.parametrize("statements1,statements2,expected", [
    ([Statement(0, Relation.LEQ, 5)], [Statement(0, Relation.MT, 5)], False)
])
def test_statements_overlap(statements1: List[Statement], statements2: List[Statement], expected: bool):
    assert statements_overlap(statements1, statements2) == expected
    assert statements_overlap(statements2, statements1) == expected
