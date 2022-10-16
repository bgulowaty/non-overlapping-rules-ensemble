# non-overlapping-rules-ensemble

Source code for paper [Extracting Interpretable Decision Tree Ensemble from Random Forest](https://ieeexplore.ieee.org/document/9533601) by [B.Gulowaty](https://www.researchgate.net/profile/Bogdan-Gulowaty) and [M.Woźniak](https://www.researchgate.net/profile/Michal-Wozniak-6).


## Building the package
Project uses poetry as build system. To install the package for usage in your local env, simply issue `poetry install`.

## Usage
```python
from box import Box
from note.note import run

DEFAULT_PARAMS = {
    "n_estimators": 5,
    "min_samples_split": 2,
    "n_jobs": 1,
    "max_depth": 5,
    "subspaces": 5,
    "cv": 5,
    "cv_repeats": 10,
}

clf_rf = train_random_forest()

results = run(x_train, y_train, clf_rf, Box(DEFAULT_PARAMS))

print(results)
``` 

For more extensive examples of usage see [usage_example.ipynb](usage_example.ipynb) notebook.



### Parallelization

The inner working are based on [joblib](https://joblib.readthedocs.io/en/latest/) parallelization library. 
Method `run` accepts joblib's [parallel backend](https://joblib.readthedocs.io/en/latest/parallel.html#custom-backend-api-experimental) keyword argument. Default backend uses `threading`. 
