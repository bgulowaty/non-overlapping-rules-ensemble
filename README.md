# non-overlapping-rules-ensemble

Source code for paper [Extracting Interpretable Decision Tree Ensemble from Random Forest](https://ieeexplore.ieee.org/document/9533601) by [B.Gulowaty](https://www.researchgate.net/profile/Bogdan-Gulowaty) and [M.Woźniak](https://www.researchgate.net/profile/Michal-Wozniak-6).

## Usage

Work in progress

```python
# todo
``` 

* Method `run` returns list of models. If, during the optimization process, pareto front was created, then the list contains all models based on the pareto front solutions. Otherwise list constains just one model. 
* Retured models are compatible with Sklearn API



### Parallelization

The inner working are based on [joblib](https://joblib.readthedocs.io/en/latest/) parallelization library. 
Method `xxx` accepts joblib's [parallel backend](https://joblib.readthedocs.io/en/latest/parallel.html#custom-backend-api-experimental) keyword argument. Default backend uses `threading`. 
