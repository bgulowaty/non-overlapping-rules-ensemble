import tempfile

import networkx as nx
import numpy as np
import scipy
from box import Box
from imblearn.metrics import geometric_mean_score
from joblib import Parallel, delayed, Memory
from networkx.algorithms.clique import find_cliques
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import make_scorer, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sympy import parse_expr
from toolz.curried import pipe, filter, map, reduce

from note.v2.rule_extractor import get_rules, measure_rules
from note.v2.rule_subspace_classifier import to_subspace_classifier
from mlutils.scikit.competence_region_ensemble import SimpleCompetenceRegionEnsembleV2

memory = Memory(tempfile.mkdtemp(), verbose=3)

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

def accuracy_with_rf(clf_rf):

    def scorer(estimator, X, y):
        y_rf = clf_rf.predict(X)
        y_model = estimator.predict(X)

        return accuracy_score(y_rf, y_model)

    return scorer

def bal_accuracy_with_rf(clf_rf):
    def scorer(estimator, X, y):
        print("balacc")
        print(y)
        y_rf = clf_rf.predict(X)
        print(y_rf)
        y_model = estimator.predict(X)
        return balanced_accuracy_score(y_rf, y_model)

    return scorer

def train_model(rules, x_train, y_train, max_depth):
    labels = [f'subspace{i}' for i in range(len(rules))]
    subspace_clf = to_subspace_classifier(x_train, y_train, rules, labels, "default")

    default_dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    default_dt.fit(x_train, y_train)

    ensemble = SimpleCompetenceRegionEnsembleV2(subspace_clf,
        {
        **{label: DecisionTreeClassifier(random_state=42, max_depth=max_depth) for label in labels},
        'default': default_dt
    })

    ensemble.fit(x_train, y_train)

    return ensemble

def score_for_rules(rules, x_train, y_train, cv, max_depth, selection_methods, clf_rf, x_test = None, y_test = None):
    clf = train_model(rules, x_train, y_train, max_depth)
    skf = ShuffleSplit(n_splits=cv, test_size=0.5, random_state=42)

    scores = cross_validate(clf, x_train, y_train, scoring={
        'balanced_accuracy': 'balanced_accuracy',
        'f1': 'f1_weighted',
        'accuracy': 'accuracy',
        'g_mean': make_scorer(geometric_mean_score, average='weighted'),
        'recall': 'recall_weighted',
        'precision': 'precision_weighted',
        'rf_accuracy': accuracy_with_rf(clf_rf),
        'rf_balanced_accuracy': bal_accuracy_with_rf(clf_rf),
    }, cv=skf, error_score='raise')

    if x_test is not None:
        clf_test_predictions = clf.predict(x_test)
        clf_rf_test_predictions = clf_rf.predict(x_test)
        test_scores = {
            'test_balanced_accuracy': balanced_accuracy_score(y_test, clf_test_predictions),
            'test_f1': f1_score(y_test, clf_test_predictions, average='weighted'),
            'test_accuracy': accuracy_score(y_test, clf_test_predictions),
            'test_g_mean': geometric_mean_score(y_test, clf_test_predictions, average='weighted'),
            'test_recall': recall_score(y_test, clf_test_predictions, average='weighted'),
            'test_precision': precision_score(y_test, clf_test_predictions, average='weighted'),
            'test_rf_accuracy': accuracy_score(clf_test_predictions, clf_rf_test_predictions),
            'test_rf_balanced_accuracy': balanced_accuracy_score(clf_test_predictions, clf_rf_test_predictions),
        }
    else:
        test_scores = {}

    scores_without_test_preffix = {
        **{k[5:] if k.startswith('test_') else k: np.mean(v) for k, v in scores.items()},
        **{f"{k[5:]}_stddev" if k.startswith('test_') else f"{k}_stddev": np.std(v) for k, v in scores.items()},
    }
    score_by_selection_method = {
        'score ' + method: float(parse_expr(method).evalf(subs=scores_without_test_preffix)) for method in
        selection_methods
    }

    return {
        **scores_without_test_preffix,
        **score_by_selection_method,
        **test_scores
    }

def run(x_train, y_train, clf_rf, params, x_test = None, y_test = None):
    all_rules = pipe(
        clf_rf.estimators_,
        map(lambda estimator: get_rules(estimator)),
        reduce(tuple.__add__),
        set,
        list
    )
    print(f"Rules={len(all_rules)}")

    print("Measuring rules")
    rule_measurements = measure_rules(all_rules, n_jobs=params.n_jobs)

    print("Adding rules to graph")
    g = nx.Graph()
    for (x, y), measurement in rule_measurements.items():
        if measurement == False:
            x_idx = all_rules.index(x)
            y_idx = all_rules.index(y)
            g.add_node(x_idx)
            g.add_node(y_idx)
            g.add_edge(x_idx, y_idx)

    print("Finding cliques")
    all_subspaces = list(filter(lambda s: len(s) <= params.subspaces)(find_cliques(g)))
    print(f"Cliques found: {len(all_subspaces)}")

    subspaces_to_check = all_subspaces

    for clique_size in reversed(range(1, params.subspaces + 1)):
        subspaces_by_size_of_param = list(filter(lambda s: len(s) == params.subspaces)(all_subspaces))
        print(f"clique = {clique_size}")
        if not len(subspaces_by_size_of_param) == 0:
            subspaces_to_check = subspaces_by_size_of_param
            break

    if not subspaces_to_check:
        subspaces_to_check = [[r_idx] for r_idx in list(range(len(all_rules)))]

    print("Scoring subpaces")

    print(subspaces_to_check)
    score_by_subspace = \
        dict(zip(
            map(tuple)(subspaces_to_check),
            Parallel(n_jobs=1)(
                delayed(lambda subspace: score_for_rules([all_rules[i] for i in subspace], x_train, y_train, params.cv, params.max_depth, params.selection_methods, clf_rf, x_test=x_test, y_test=y_test))(subspace)
                for subspace in subspaces_to_check)
        ))


    best_by_train_acc = max(score_by_subspace.values(), key=lambda it: it['accuracy'])

    best_by_test_acc = {}
    if x_test is not None:
        best_by_test_acc = max(score_by_subspace.values(), key=lambda it: it['test_accuracy'])

    scores_by_selection_method = {}
    for selection_method in params.selection_methods:
        best_score = max([subspace[f'score {selection_method}'] for subspace in score_by_subspace.values()])
        best_score_rules = [rules for rules, val in score_by_subspace.items() if val[f'score {selection_method}'] == best_score]

        best_score_rules_with_scoring = {
            rules: score_by_subspace[rules] for rules in best_score_rules
        }

        worst = best_score_rules_with_scoring[min(best_score_rules_with_scoring, key=lambda v: best_score_rules_with_scoring[v]['accuracy'])]
        best = best_score_rules_with_scoring[max(best_score_rules_with_scoring, key=lambda v: best_score_rules_with_scoring[v]['accuracy'])]
        if x_test is not None:
            best_on_test = best_score_rules_with_scoring[min(best_score_rules_with_scoring, key=lambda v: best_score_rules_with_scoring[v]['test_accuracy'])]
        else:
            best_on_test = {}

        scores_by_selection_method[selection_method] = {
            **{f'worst_{k}': v for k, v in worst.items()},
            **{f'best_{k}': v for k, v in best.items()},
            **{f'best_test_{k}': v for k, v in best_on_test.items()},
            'found': len(best_score_rules)
        }

    return {
        **scores_by_selection_method,
        "best_by_train_acc": best_by_train_acc,
        "best_by_test_acc": best_by_test_acc
    }
#%%

