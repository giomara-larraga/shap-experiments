# Shapley values in multiobjective optimization
## Installation
To install the project, clone it and install it using `poetry`:

```
$ git clone git@github.com:gialmisi/shap-experiments.git
$ cd shap-experiments
$ poetry install
```

## Notebooks
There are some notebooks in `notebooks/` to play around with. If you installed the project using `poetry`, you can run `jupyter-lab` to launch a Jupyter lab session in your web browser to
access the notebooks. You might need to install a custom kernel so that Jupyter is able to pick up on the environemnt created by `poetry`. To add a custom kernel named _SHAP_, issue
the follwoing command:

```
$ python -m ipykernel install --user --name SHAP
```

Remember to select the right kernel in the Jupyter lab environment!

