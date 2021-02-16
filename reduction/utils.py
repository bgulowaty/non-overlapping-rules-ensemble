import numpy as np
from collections import defaultdict
from typing import Set, Collection

from toolz.curried import pipe, filter, map, reduce

from .api import Rule, Relation, Statement
from .classification import to_instance


EPS = np.finfo(float).eps

def coverage_by_rule(rules, x):
    covered_instances_by_rule = defaultdict(lambda: 0)
    for rule in rules:
        for x_single in x:
            x_inst = to_instance(x_single)
            if rule.describes(x_inst):
                covered_instances_by_rule[rule] = covered_instances_by_rule[rule] + 1

    return covered_instances_by_rule

def join_consecutive_statements(rule: Rule) -> Rule:
    all_statements = set()

    for feature in rule.get_features():
        statements = rule.get_statements_for_feature(feature)
        new_this_rule_statements = set(statements)

        sorted_by_threshold = iter(sorted(statements, key=lambda s: s.threshold))
        try:
            current_statement = next(sorted_by_threshold)
            while True:
                next_statement = next(sorted_by_threshold)

                if current_statement.relation == next_statement.relation:
                    if current_statement.relation == Relation.MT:
                        new_this_rule_statements.remove(next_statement)
                    else:
                        new_this_rule_statements.remove(current_statement)

                current_statement = next_statement
        except StopIteration:
            pass

        all_statements = all_statements.union(new_this_rule_statements)

    return Rule(all_statements, rule.distribution_or_class)


def bound_if_needed(rule: Rule, statement: Statement, feature_min_values, feature_max_values) -> Set[Statement]:
    statements = set(rule.get_statements_for_feature(statement.feature_idx)).difference({statement})
    new_statements = set()
    next_higher = pipe(
        statements,
        filter(lambda s: s.threshold > statement.threshold),
        list
    )

    next_lower = pipe(
        statements,
        filter(lambda s: s.threshold < statement.threshold),
        list
    )

    if statement.relation == Relation.LEQ and not any(next_lower):
        new_statements.add(Statement(statement.feature_idx, Relation.MT, feature_min_values[statement.feature_idx]))
    elif statement.relation == Relation.MT and not any(next_higher):
        new_statements.add(Statement(statement.feature_idx, Relation.LEQ, feature_max_values[statement.feature_idx]-EPS))
    else:
        return set()

    return new_statements

def bound_rule(rule: Rule, x_train) -> Rule:
    feature_max_values = np.max(x_train, axis=0)
    feature_min_values = np.min(x_train, axis=0)

    new_statements_for_rule = set(rule.statements)
    for statement in rule.statements:
        new_statements_for_rule = new_statements_for_rule.union(
            bound_if_needed(rule, statement, feature_min_values, feature_max_values))

    for feature_idx in range(len(feature_max_values)):
        statements_for_feature = rule.get_statements_for_feature(feature_idx)

        if len(statements_for_feature) == 0:
            lower_statement = Statement(feature_idx, Relation.MT, feature_min_values[feature_idx]-EPS)
            upper_statement = Statement(feature_idx, Relation.LEQ, feature_max_values[feature_idx])
            new_statements_for_rule = new_statements_for_rule.union({lower_statement, upper_statement})

    return Rule(new_statements_for_rule, rule.distribution_or_class)


# Rules must not be adjacent
def calculate_coverage(rules: Collection[Rule], x_train):
    feature_ranges = np.ptp(x_train, axis=0)
    total_area = 0
    max_area = reduce(lambda a,b: a*b)(feature_ranges)
    for rule in rules:
        bounded_rule = bound_rule(rule, x_train)
        this_rule_statement_ranges = []
        for feature, statements in bounded_rule.get_statements_by_feature().items():
            sorted_thresholds = pipe(statements,
                 map(lambda s: s.threshold),
                 sorted,
                 list
                 )

            this_rule_statement_ranges.append(np.abs(sorted_thresholds[-1] - sorted_thresholds[0]))

        total_area = total_area + reduce(lambda a,b: a*b)(this_rule_statement_ranges)

    return 1 if total_area/max_area > 1 else total_area/max_area

import numpy as np


def flip_random_n_elements_in_vector(vec, n: int):
    available_elements = np.unique(vec)

    get_other_element_than = lambda el: np.random.choice(list(set(available_elements).difference({el})))

    to_flip = np.random.choice(range(len(vec)), size=n, replace=False)

    return [
        val if idx not in to_flip else get_other_element_than(val) for idx, val in enumerate(vec)
    ]

# def test_calculate_coverate():
#     rule = Rule([
#         Statement
#     ])

def test_join_consecutive():
    rule = Rule([
        Statement(0, Relation.MT, 10),
        Statement(0, Relation.MT, 20),
        Statement(0, Relation.LEQ, 35),
        Statement(0, Relation.LEQ, 40),
        Statement(0, Relation.LEQ, 50),
        Statement(1, Relation.MT, 10),
        Statement(1, Relation.MT, 20),
    ], 1)

    actual = join_consecutive_statements(rule)
    assert set(actual.statements) == {
        Statement(0, Relation.MT, 10),
        Statement(0, Relation.LEQ, 50),
        Statement(1, Relation.MT, 10),
    }

def test_bound_rule():
    rule = Rule([
        Statement(0, Relation.MT, 0),
        Statement(1, Relation.LEQ, 20),
    ], 1)

    x_train = [
        [-5, -5], [30, 30]
    ]

    bounded_rule = bound_rule(rule, x_train)

    assert set(bounded_rule.statements) == {
        Statement(0, Relation.MT, 0),
        Statement(0, Relation.LEQ, 30),
        Statement(1, Relation.LEQ, 20),
        Statement(1, Relation.MT, -5-EPS)
    }

def test_bound_rule_2():
    rule = Rule([
        Statement(0, Relation.MT, 0),
        Statement(0, Relation.LEQ, 5),
        Statement(1, Relation.LEQ, 20),
    ], 1)

    x_train = [
        [-5, -5], [30, 30]
    ]

    bounded_rule = bound_rule(rule, x_train)

    assert set(bounded_rule.statements) == {
        Statement(0, Relation.MT, 0),
        Statement(0, Relation.LEQ, 5),
        Statement(1, Relation.LEQ, 20),
        Statement(1, Relation.MT, -5-EPS)
    }

def test_bound_rule_3():
    rule = Rule([
        Statement(0, Relation.MT, 0),
        Statement(0, Relation.LEQ, 5),
    ], 1)

    x_train = [
        [-5, -5], [30, 30]
    ]

    bounded_rule = bound_rule(rule, x_train)

    print(bounded_rule.statements)

    assert set(bounded_rule.statements) == {
        Statement(0, Relation.MT, 0),
        Statement(0, Relation.LEQ, 5),
        Statement(1, Relation.LEQ, 30),
        Statement(1, Relation.MT, -5-EPS)
    }
