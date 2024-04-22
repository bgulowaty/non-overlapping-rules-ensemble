from functools import reduce

import numpy as np
from box import Box
from sklearn.utils import check_array
from sympy import parse_expr, reduce_inequalities, lambdify


def to_subspace_classifier(x_train, y_train, rules, labels, non_matching_label = None):
    rules_as_sympy_inequalities = [
        reduce_inequalities([parse_expr(f"f{statement[0]} {statement[1]} {statement[2]}") for statement in rule]) for rule in rules
    ]
    all_variables = reduce(lambda x, y: x.union(y), [it.free_symbols for it in rules_as_sympy_inequalities])
    all_variables_as_str = [str(it) for it in all_variables]
    sorted_variables = sorted(all_variables_as_str, key=lambda it: int(it[1:]))

    predicate_by_rule = {
        label: lambdify(sorted_variables, rule) for label, rule in zip(labels, rules_as_sympy_inequalities)
    }

    x_train = check_array(x_train)
    x_train_adjusted_to_features_count = x_train[:, :len(sorted_variables)]


    samples_covered_by_subspace = {}

    for label, predicate in predicate_by_rule.items():
        covered_by_predicate = np.apply_along_axis(lambda row: predicate(*row), 1, x_train_adjusted_to_features_count)
        samples_covered_by_subspace[label] = covered_by_predicate.sum()


    samples_covered_by_default_tree = len(x_train) - sum(samples_covered_by_subspace.values())
    if samples_covered_by_default_tree < 10:
        biggest_subspace = max(samples_covered_by_subspace, key=samples_covered_by_subspace.get)
        non_matching_label = biggest_subspace

    print(f"Samples covered by default tree = {samples_covered_by_default_tree}")
    print(f"Non matching label = {non_matching_label}")

    predicate_by_rule = {k: v for k,v in predicate_by_rule.items() if samples_covered_by_subspace[k] >= 10}
    print(predicate_by_rule)
    print(samples_covered_by_subspace)
    def predict(x, *args, **kwargs):
        x = check_array(x)
        x_adjusted_to_features_count = x[:, :len(sorted_variables)]

        predictions = np.full(len(x), False, dtype=object)

        for label, predicate in predicate_by_rule.items():
            print(f"Label {label}")
            is_covered_by_this_rule = np.apply_along_axis(lambda row: predicate(*row), 1, x_adjusted_to_features_count)

            predictions[is_covered_by_this_rule] = label

        predictions_without_label = predictions == False
        if np.any(predictions_without_label):
            if not non_matching_label:
                raise Exception("There are samples not covered by rules and no default label was given")

            predictions[predictions_without_label] = non_matching_label
        print(np.unique(predictions))
        return predictions

    return Box({
        'predict': predict
    })


def test_rule_subspace_classifier():
    rules = [
        [[0, '>', 5], [1, '<', 0]],
        [[0, '<', -5], [1, '>', 5]]
    ]
    labels = ['a', 'b']

    clf = to_subspace_classifier(rules, labels, non_matching_label= 'c')

    labels = clf.predict([
        [10, -5],
        [-10, 10],
        [0, 3],
    ])

    assert np.array_equal(labels, ['a', 'b', 'c'])