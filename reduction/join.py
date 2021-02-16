from collections import defaultdict
from itertools import groupby, product
from typing import Optional, Set

import pytest

from .api import Rule, AdjacentOrNot, Statement, Relation
from .measure_adjacencies import calculate_span_for_statements, Span


def join_if_possible(rule1: Rule, rule2: Rule) -> Optional[Rule]:
    spans_by_feature_idx_1 = {feature: calculate_span_for_statements(statements) for feature, statements in
                              groupby(rule1.statements, lambda statement: statement.feature_idx)}
    spans_by_feature_idx_2 = {feature: calculate_span_for_statements(statements) for feature, statements in
                              groupby(rule2.statements, lambda statement: statement.feature_idx)}

    overlaps = any([any_span_overlaps(spans1, spans2) for feature1, spans1 in spans_by_feature_idx_1.items()
                    for feature2, spans2 in spans_by_feature_idx_2.items() if feature1 == feature2])

    if overlaps:
        return None

    new_distribution = defaultdict(lambda: 0)
    for label, value in rule1.distribution_or_class.items():
        new_distribution[label] = new_distribution[label] + value
    for label, value in rule2.distribution_or_class.items():
        new_distribution[label] = new_distribution[label] + value

    return Rule(rule1.statements + rule2.statements, new_distribution)


def any_span_overlaps(spans1: Set[Span], spans2: Set[Span]) -> bool:
    return any([span1.overlaps(span2) for span1, span2 in product(spans1, spans2)])

    if adjacency == AdjacentOrNot.ADJACENT and rule1.classified_class != rule2.classified_class:
        return None

    return Rule(rule1.statements + rule2.statements, rule1.classified_class)


@pytest.mark.parametrize("rule1,rule2,expected", [
    (
            Rule([Statement(0, Relation.MT, 0), Statement(0, Relation.LEQ, 5)], 1),
            Rule([Statement(1, Relation.MT, 0), Statement(1, Relation.LEQ, 5)], 1),
            Rule([Statement(0, Relation.MT, 0), Statement(0, Relation.LEQ, 5),
                  Statement(1, Relation.MT, 0), Statement(1, Relation.LEQ, 5)], 1)
    ),
    (
            Rule([Statement(0, Relation.MT, 0)], 1),
            Rule([Statement(1, Relation.MT, 0)], 2),
            None
    ),
    (
            Rule([Statement(0, Relation.MT, 0), Statement(0, Relation.LEQ, 5), Statement(1, Relation.MT, 10)], 1),
            Rule([Statement(0, Relation.MT, -5), Statement(0, Relation.LEQ, 5), Statement(1, Relation.LEQ, 0)], 1),
            None
    ),
])
def test_joins(rule1, rule2, expected):
    assert join_if_possible(rule1, rule2) == expected
