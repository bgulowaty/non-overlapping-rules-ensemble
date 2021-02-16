from __future__ import annotations

from collections import defaultdict, OrderedDict
from typing import Collection, Set, Callable, Dict, List

import numpy as np
from attr import attrs, attrib
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y
from toolz.curried import map, pipe, filter

from .api import Rule, Instance


def to_instance(x: any):
    return Instance(dict(enumerate(x)))


def classify_single_value(x: Instance, rules: Collection[Rule]) -> any:
    supports = defaultdict(lambda: 0)

    for rule in rules:
        if rule.describes(x):
            supports[rule.classified_class] = supports[rule.classified_class] + 1

    if not supports:
        return None
    return max(supports, key=supports.get)


@attrs()
class SubspaceRulesClassifier(BaseEstimator):
    rules: List[Rule] = attrib()
    max_depth: int = attrib()
    random_state: int = attrib(default=42)
    train_default_on_whole_set: bool = attrib(default=True)
    _clf_by_rule: Dict[Rule, any] = attrib(init=False, factory=dict)
    _default_clf: any = attrib(init=False)

    def fit(self, x, y):
        x, y = check_X_y(x, y)

        already_covered_indicies = set()

        for rule in self.rules:
            depth = self.max_depth - len(rule.statements)

            matching_sample_indicies = pipe(
                x,
                map(to_instance),
                enumerate,
                filter(lambda idx_with_instance: rule.describes(idx_with_instance[1])),
                map(lambda idx_with_instance: idx_with_instance[0]),
                list
            )

            if depth >= 1 and len(matching_sample_indicies) >= 1:

                if any(already_covered_indicies.intersection(set(matching_sample_indicies))):
                    raise Exception("Rules are overlapping")

                already_covered_indicies.update(matching_sample_indicies)

                x_train = x[matching_sample_indicies]
                y_train = y[matching_sample_indicies]

                clf = DecisionTreeClassifier(random_state=self.random_state, max_depth=depth)
                clf.fit(x_train, y_train)
                self._clf_by_rule[rule] = clf

        not_covered_indices = list(set(range(len(x))).difference(already_covered_indicies))
        default_clf = DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth)
        if len(not_covered_indices) == 0 or self.train_default_on_whole_set:
            default_clf.fit(x, y)
        else:
            x_train = x[not_covered_indices]
            y_train = y[not_covered_indices]
            default_clf.fit(x_train, y_train)

        self._default_clf = default_clf

    def predict(self, x):

        instance_idx_by_rule = defaultdict(list)
        not_covered_indicies = set(range(len(x)))
        for rule in self._clf_by_rule.keys():
            for idx, single_x in enumerate(x):
                if rule.describes(to_instance(single_x)):
                    instance_idx_by_rule[rule].append(idx)
                    not_covered_indicies.remove(idx)

        label_by_idx = OrderedDict()
        for rule, indicies in instance_idx_by_rule.items():
            clf = self._clf_by_rule[rule]
            x_to_classify = x[indicies]

            for label, idx in zip(clf.predict(x_to_classify), indicies):
                label_by_idx[idx] = label

        x_indicies_not_covered = list(not_covered_indicies)

        if len(x_indicies_not_covered) > 0:
            x_not_covered_to_classify = x[x_indicies_not_covered]

            for label, idx in zip(self._default_clf.predict(x_not_covered_to_classify), x_indicies_not_covered):
                label_by_idx[idx] = label

        return list(label_by_idx.values())
    #
    # def predict(self, x):
    #     return pipe(
    #         x,
    #         map(self._classify_instance),
    #         list
    #     )

    def _classify_instance(self, x: List[any]):
        clf = self._find_corresponding_classifier(to_instance(x))

        return clf.predict([x])[0]

    def _find_corresponding_classifier(self, x: Instance) -> any:
        matching_rules = pipe(
            self._clf_by_rule.keys(),
            filter(lambda r: r.describes(x)),
            list
        )

        if len(matching_rules) > 1:
            raise Exception("Too many matching rules")

        if len(matching_rules) == 0:
            return self._default_clf

        return self._clf_by_rule[matching_rules[0]]


@attrs(frozen=True)
class RulesClassifier:
    rules: Set[Rule] = attrib(converter=frozenset)
    classification_strategy: Callable[[Instance, Collection[Rule]], any] = attrib(default=classify_single_value)

    def fit(self, x, y):
        raise NotImplementedError("aww")

    def predict(self, x):
        y_pred = []
        for inst in x:
            classified_class = self.classification_strategy(to_instance(inst), self.rules)

            y_pred.append(classified_class)

        return y_pred


def measure_acc(x, y, rules) -> float:
    y_new = RulesClassifier(rules).predict(x)

    correct_classification = sum([1 for y_pred, y_actual in zip(y_new, y) if y_pred == y_actual])

    return correct_classification / len(y)


def measure_metric(x, y, rules, metric) -> float:
    if len(rules) == 0:
        return 0

    y_new = RulesClassifier(rules).predict(x)

    labels = set(np.unique(y))
    get_label_other_than = lambda current_label: list(labels.difference({current_label}))[0]

    y_new_with_labels = [
        y if y is not None else get_label_other_than(y) for y in y_new
    ]

    return metric(y, y_new_with_labels)
