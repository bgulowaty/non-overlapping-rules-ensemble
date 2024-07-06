
import pandas as pd
from box import Box
import pprint
from sklearn.ensemble import RandomForestClassifier
from .note_v2 import run

def test_passes_for_default_arguments_and_breast_cancer():
    train_data = pd.read_csv('../breast-train-0-s1.csv')
    x_train = train_data.drop('TARGET', axis=1).values
    y_train = train_data['TARGET'].values

    test_data = pd.read_csv('../breast-test-0-s1.csv')
    x_test = test_data.drop('TARGET', axis=1).values
    y_test = test_data['TARGET'].values

    params = Box({
        "subspaces": 5,
        "n_jobs": 8,
        "selection_methods": ['balanced_accuracy', 'accuracy', 'rf_accuracy', 'rf_balanced_accuracy', 'accuracy/accuracy_stddev'],
        "cv": 5,
        "max_depth": 5
    })


    rf = RandomForestClassifier(n_estimators=5, max_depth=5)
    rf.fit(x_train, y_train)

    results = run(x_train, y_train, rf, params, x_test=x_test, y_test=y_test)
    pprint.pprint(results)


def test_note_v2_without_test_data():
    from sklearn.datasets import load_iris
    rf = RandomForestClassifier(n_estimators=5, random_state=42, max_depth=None)
    data = load_iris()
    x, y = data.data, data.target
    rf.fit(x, y)

    params = Box({
        "subspaces": 3,
        "n_jobs": 8,
        "selection_methods": ['balanced_accuracy', 'accuracy', 'rf_accuracy', 'rf_balanced_accuracy', 'accuracy/accuracy_stddev'],
        "cv": 5,
        "depth": 5
    })

    results = run(x, y, rf, params)
    pprint.pprint(results)

#%%