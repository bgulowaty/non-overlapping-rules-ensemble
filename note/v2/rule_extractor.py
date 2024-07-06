import contextlib
from functools import reduce
from itertools import combinations, groupby

import joblib
import sympy
from joblib import delayed, Parallel
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.tree._tree import Tree
from sympy import reduce_inequalities, And
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm


def get_rules(sklearn_dt: DecisionTreeClassifier, return_class_distributions=False) -> set:
    tree_: Tree = sklearn_dt.tree_

    def recurse(node: int, statements=tuple()) -> set:
        this_node_feature_idx: int = tree_.feature[node]

        if this_node_feature_idx != _tree.TREE_UNDEFINED:
            threshold: float = tree_.threshold[node]

            left_statements = statements + (
                (this_node_feature_idx, "<=", threshold),
            )
            left_rules = recurse(tree_.children_left[node], left_statements)

            right_statements = statements + (
                (this_node_feature_idx, ">", threshold),
            )
            right_rules = recurse(tree_.children_right[node], right_statements)
            return left_rules + right_rules
        else:  # reached terminal leaf
            samples_count_for_each_class = tree_.value[node][0]
            samples_count_by_class = {
                sklearn_dt.classes_[idx]: count
                for idx, count in enumerate(samples_count_for_each_class)
            }
            return (statements,)

    return recurse(0)


def rule_overlaps(rule1, rule2):
    all_statements = rule1 + rule2

    inequalities = [
        parse_expr(f"f{statement[0]} {statement[1]} {statement[2]}") for statement in all_statements
    ]

    return reduce_inequalities(inequalities) != False  # "It has solution"

def rule_overlaps_2(rule1, rule2):
    return any({k: reduce(lambda x,y: x.intersect(y), [parse_expr(f"f{statement[0]} {statement[1]} {statement[2]}").as_set() for statement in v]) != sympy.EmptySet
         for k,v in groupby(sorted(rule1 + rule2, key= lambda i: i[0]), lambda i: i[0])} \
        .values())



@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def measure_rules(all_rules, n_jobs: int = 1):
    rule_combinations = list(combinations(all_rules, 2))
    print(f"Generated {len(rule_combinations)} combinations")

    with tqdm_joblib(tqdm(desc="measure_rules", total=len(rule_combinations))) as progress_bar:
        measured_combinations = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(
                lambda comb: (comb, rule_overlaps(comb[0], comb[1]))
            )(combination)
            for combination in rule_combinations
        )

    return dict(measured_combinations)

def measure_rules_2(all_rules, n_jobs: int = 1):
    rules_as_inequalities = [
        [parse_expr(f"f{statement[0]} {statement[1]} {statement[2]}") for statement in rule] for rule in all_rules
    ]

    combination_idxes = list(combinations(range(len(all_rules)), 2))

    rule_combinations = [
        (rules_as_inequalities[i], rules_as_inequalities[j]) for (i, j) in combination_idxes
    ]

    print(f"Generated {len(rule_combinations)} combinations")

    with tqdm_joblib(tqdm(desc="measure_rules", total=len(rule_combinations))) as progress_bar:
        measured_combinations = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(
                lambda comb: not reduce_inequalities(comb[0] + comb[1])
            )(combination)
            for combination in rule_combinations
        )
        print("HALO")
    print("Zipping!:<")
    return dict(zip(combination_idxes, measured_combinations))

def to_set_by_feature(rule):
    return {k: [parse_expr(f"f{statement[0]} {statement[1]} {statement[2]}").as_set() for statement in v]
     for k,v in groupby(sorted(rule, key= lambda i: i[0]), lambda i: i[0])}

def spans_overlap(spans1, spans2):
    all_features = set(list(spans1.keys()) + list(spans2.keys()))

    return any(reduce(lambda x,y: x.intersect(y), spans1.get(feature, sympy.EmptySet) + spans2.get(feature, sympy.UniversalSet)) != sympy.EmptySet
               for feature in all_features)


def test_measurer_adjacent():
    rule1 = [
        [0, ">", 1],
        [0, "<=", 3],
    ]

    rule2 = [
        [1, ">", 10],
        [1, "<=", 20],
    ]

    assert rule_overlaps(rule1, rule2) == True
    assert rule_overlaps(rule2, rule1) == True

def test_measurer_adjacent_2():
    rule1 = [
        [0, ">", 1],
        [1, "<=", 3],
    ]

    rule2 = [
        [0, "<=", 1],
        [1, ">", 3],
    ]

    assert rule_overlaps(rule1, rule2) == False
    assert rule_overlaps(rule2, rule1) == False


def test_measurer_not_adjacent():
    rule1 = [
        [0, ">", 1],
        [0, "<=", 3],
        [1, "<=", 3],
    ]

    rule2 = [
        [0, ">", 10],
        [0, "<=", 20],
        [1, ">", 5],
        [1, "<=", 10],
    ]

    assert rule_overlaps(rule1, rule2) == False
    assert rule_overlaps(rule2, rule1) == False


def test_measurer_adjacent_2():
    rule1 = [
        [0, ">", 1],
        [0, "<=", 3],
        [2, ">", 1],
        [2, "<=", 3],
    ]

    rule2 = [
        [0, ">", 1],
        [0, "<=", 3],
        [1, ">", 1],
        [1, "<=", 3],
    ]

    assert rule_overlaps(rule1, rule2) == True
    assert rule_overlaps(rule2, rule1) == True


def test_measurer_adjacent_real():
    rule1 = [
        [3, "<=", 2.5],
        [2, ">", 1.1],
        [0, ">", 5.75],
        [1, "<=", 3.700000047683716],
        [0, "<=", 7.9],
        [2, "<=", 4.950000047683716],
        [3, ">", 1.699999988079071],
        [1, ">", 2.0],
    ]
    rule2 = [
        [3, "<=", 2.5],
        [2, "<=", 6.9],
        [2, ">", 2.350000023841858],
        [3, ">", 1.6500000357627869]
    ]

    assert rule_overlaps(rule1, rule2) == True
    assert rule_overlaps(rule2, rule1) == True
