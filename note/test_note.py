
import pandas as pd
from box import Box
from sklearn.ensemble import RandomForestClassifier

from note.note import run, DEFAULT_PARAMS
def test_passes_for_default_arguments_and_breast_cancer():
    train_data = pd.read_csv('breast-train-0-s1.csv')
    x_train = train_data.drop('TARGET', axis=1).values
    y_train = train_data['TARGET'].values

    rf = RandomForestClassifier(n_estimators=5, max_depth=5)
    rf.fit(x_train, y_train)

    run(x_train, y_train, rf, Box(DEFAULT_PARAMS))