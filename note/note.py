
import time
import warnings
from collections import defaultdict

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from joblib import Parallel, delayed
from networkx.algorithms.clique import find_cliques
from rules.api import AdjacentOrNot
from rules.classification.rule_measures import BayesianRuleMeasures, covered_by_statements
from rules.classification.subspace_rules_classifier import SubspaceRulesClassifier
from rules.note.extract_rules import extract_rules
from rules.note.overlapping.measure_adjacencies import measure_rules
from rules.utils.utils import join_consecutive_statements
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, make_scorer, confusion_matrix, f1_score, precision_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sympy import parse_expr
from toolz.curried import pipe, filter, map, reduce

DEFAULT_PARAMS = {
    "n_estimators": 5,
    "min_samples_split": 2,
    "n_jobs": 1,
    "max_depth": 5,
    "subspaces": 5,
    "cv": 5,
    "cv_repeats": 10,
    "selection_methods": ['balanced_accuracy', 'accuracy', 'rf_accuracy', 'rf_balanced_accuracy', 'accuracy/accuracy_stddev'] # 'balanced_accuracy' # 'accuracy', 'f1_weighted'
}


def run(x_train, y_train, clf_rf, params):

    all_rules = pipe(
        clf_rf.estimators_,
        map(lambda estimator: extract_rules(estimator)),
        reduce(set.union),
        map(lambda r: join_consecutive_statements(r)),
        #     map(lambda r: bound_rule(r, x_train)),
        list
    )

    all_statements = pipe(
        all_rules,
        set,
        map(lambda r: list(r.statements)),
        reduce(list.__add__),
    )

    def add_to_graph(measurements_tuple, measurement):

        if measurement == AdjacentOrNot.NOT_ADJACENT:
            rule_1, rule_2 = measurements_tuple
            rule_idx_1 = all_rules.index(rule_1)
            rule_idx_2 = all_rules.index(rule_2)

            g.add_node(rule_idx_1)

            g.add_node(rule_idx_2)

            g.add_edge(rule_idx_1, rule_idx_2)

    g = nx.Graph()

    with joblib.parallel_backend('threading'):
        all_rule_measurements = measure_rules(all_rules, n_jobs=params.n_jobs)


    for measurements_tuple, measurement in all_rule_measurements.items():
        add_to_graph(measurements_tuple, measurement)

    all_subspaces = list(filter(lambda s: len(s) <= params.subspaces)(find_cliques(g)))

    subspaces_to_check = all_subspaces

    for clique_size in reversed(range(1, params.subspaces + 1)):
        subspaces_by_size_of_param = list(filter(lambda s: len(s) == params.subspaces)(all_subspaces))
        print(f"clique = {clique_size}")
        if not len(subspaces_by_size_of_param) == 0:
            subspaces_to_check = subspaces_by_size_of_param
            break

    if not subspaces_to_check:
        subspaces_to_check = [[r_idx] for r_idx in list(range(len(all_rules)))]

    classes_count = len(np.unique(y_train))

    def calculate_entropy(y, classes_count):
        counts = np.unique(y, return_counts=True)[1]
        return entropy(counts, base=classes_count)

    def accuracy_with_rf(estimator, X, y):
        y_rf = clf_rf.predict(X)
        y_model = estimator.predict(X)

        return accuracy_score(y_rf, y_model)

    def bal_accuracy_with_rf(estimator, X, y):
        y_rf = clf_rf.predict(X)
        y_model = estimator.predict(X)

        return balanced_accuracy_score(y_rf, y_model)

    def rf_kohen_cappa(estimator, X, y):
        y_rf = clf_rf.predict(X)
        y_model = estimator.predict(X)

        return cohen_kappa_score(y_rf, y_model)

    def get_val(clique, x_train, y_train):

        rules = np.array(all_rules)[clique]
        clf = SubspaceRulesClassifier(rules=rules, max_depth=params.max_depth, random_state=42)

        skf = ShuffleSplit(n_splits=params.cv, test_size=0.5,random_state=42)
        with joblib.parallel_backend('threading'):
            scores = cross_validate(clf, x_train, y_train, n_jobs=1, scoring={
                'balanced_accuracy': 'balanced_accuracy',
                'f1': 'f1_weighted',
                'accuracy': 'accuracy',
                'g_mean': make_scorer(geometric_mean_score, average='weighted'),
                'recall': 'recall_weighted',
                'precision': 'precision_weighted',
                'rf_accuracy': accuracy_with_rf,
                'rf_balanced_accuracy': bal_accuracy_with_rf,
                'rf_cohen_kappa': rf_kohen_cappa
            }, cv=skf)

        additional_scores = defaultdict(list)

        skf = ShuffleSplit(n_splits=params.cv, test_size=0.5, random_state=42)
        for train_index, test_idx in skf.split(x_train, y_train):
            x_train_split = x_train[train_index]
            y_train_split = y_train[train_index]

            mean_rule_scores = defaultdict(list)
            for rule in rules:
                covered_indicies = list(covered_by_statements(rule, x_train_split))

                rule_measurements = BayesianRuleMeasures.create(rule, x_train_split, y_train_split)
                mean_rule_scores['a'].append(rule_measurements.a())
                mean_rule_scores['b'].append(rule_measurements.b())
                mean_rule_scores['c'].append(rule_measurements.c())
                mean_rule_scores['d'].append(rule_measurements.d())
                mean_rule_scores['s'].append(rule_measurements.s_measure())
                mean_rule_scores['n'].append(rule_measurements.n_measure())
                mean_rule_scores['entropy'].append(calculate_entropy(y_train_split[covered_indicies], classes_count))

            for score_name, this_score in dict(mean_rule_scores).items():
                additional_scores[score_name].append(np.mean(this_score))



        final_clf = SubspaceRulesClassifier(rules=rules, max_depth=params.max_depth, random_state=42)
        final_clf.fit(x_train, y_train)

        scores = {
            **scores,
            **dict(additional_scores),
        }
        scores_without_test_preffix = {
            **{k[5:] if k.startswith('test_') else k: np.mean(v) for k, v in scores.items()},
            **{f"{k[5:]}_stddev" if k.startswith('test_') else f"{k}_stddev": np.std(v) for k, v in scores.items()},
        }

        score_by_selection_method = {
            'score ' + method: float(parse_expr(method).evalf(subs=scores_without_test_preffix)) for method in params.selection_methods
        }

        return {
            **scores_without_test_preffix,
            **score_by_selection_method
        }

    with joblib.parallel_backend('threading'):
        score_by_subspace = \
            dict(zip(
                map(tuple)(subspaces_to_check),
                Parallel(n_jobs=params.n_jobs)(delayed(lambda subspace: get_val(subspace, x_train, y_train))(subspace) for subspace in subspaces_to_check)
            ))

    scores_by_selection_method = {}
    for selection_method in params.selection_methods:
        best_score = max([s[f'score {selection_method}'] for s in score_by_subspace.values()])
        best_score_rules = [rules for rules, val in score_by_subspace.items() if val[f'score {selection_method}'] == best_score]

        best_score_rules_with_scoring = {
            rules: score_by_subspace[rules] for rules in best_score_rules
        }
        worst = best_score_rules_with_scoring[min(best_score_rules_with_scoring, key=lambda v: best_score_rules_with_scoring[v]['accuracy'])]
        best = best_score_rules_with_scoring[max(best_score_rules_with_scoring, key=lambda v: best_score_rules_with_scoring[v]['accuracy'])]

        scores_by_selection_method[selection_method] = {
            **{f'worst_{k}': v for k, v in worst.items()},
            **{f'best_{k}': v for k, v in best.items()},
            'found': len(best_score_rules)
        }

    return scores_by_selection_method
