from functools import reduce

import numpy as np
from box import Box
from sklearn.utils import check_array
from sympy import parse_expr, reduce_inequalities, lambdify
from loguru import logger as log

def to_subspace_classifier(x_train, y_train, rules, labels, non_matching_label = None):
    x_train = check_array(x_train)

    rules_as_sympy_inequalities = [
        reduce_inequalities([parse_expr(f"f{statement[0]} {statement[1]} {statement[2]}") for statement in rule]) for rule in rules
    ]
    all_variables = reduce(lambda x, y: x.union(y), [it.free_symbols for it in rules_as_sympy_inequalities])
    all_variables_as_str = [str(it) for it in all_variables]
    sorted_variables = sorted(all_variables_as_str, key=lambda it: int(it[1:]))

    variables = [f"f{i}" for i in range(x_train.shape[1])]

    log.debug(f"rules = {rules}")
    log.debug(f"all vars = {all_variables_as_str}")
    log.debug(f"sorted variables = {sorted_variables}")
    log.debug(f"x tr = {x_train}")
    log.debug(f"x tr = {variables}")

    predicate_by_rule = {
        label: lambdify(variables, rule) for label, rule in zip(labels, rules_as_sympy_inequalities)
    }

    samples_covered_by_subspace = {}

    for label, predicate in predicate_by_rule.items():
        covered_by_predicate = np.apply_along_axis(lambda row: predicate(*row), 1, x_train)
        samples_covered_by_subspace[label] = covered_by_predicate.sum()

    log.debug(f"Total samples = {len(x_train)}, Total covered = {sum(samples_covered_by_subspace.values())}")
    samples_covered_by_default_tree = len(x_train) - sum(samples_covered_by_subspace.values())
    if samples_covered_by_default_tree < 10:
        biggest_subspace = max(samples_covered_by_subspace, key=samples_covered_by_subspace.get)
        log.debug(f"Non matching label based on biggest subspace! {biggest_subspace}")
        non_matching_label = biggest_subspace

    predicate_by_rule = {k: v for k,v in predicate_by_rule.items() if samples_covered_by_subspace[k] >= 10}
    log.debug(samples_covered_by_subspace)
    log.debug(predicate_by_rule)
    log.debug(f"non matching label = {non_matching_label}")

    def predict(x, *args, **kwargs):
        x = check_array(x)

        predictions = np.full(len(x), False, dtype=object)

        for label, predicate in predicate_by_rule.items():
            is_covered_by_this_rule = np.apply_along_axis(lambda row: predicate(*row), 1, x_train)

            predictions[is_covered_by_this_rule] = label

        predictions_without_label = predictions == False
        if np.any(predictions_without_label):
            if not non_matching_label:
                raise Exception("There are samples not covered by rules and no default label was given")

            predictions[predictions_without_label] = non_matching_label
        log.debug(f"predictions: {np.unique(predictions,return_counts=True)}")
        return predictions

    log.debug(f"Test prediction = {np.unique(predict(x_train), return_inverse=True)}")

    return Box({
        'predict': predict
    })


