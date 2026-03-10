# Model Registry (Student Instructions)

Use this guide to:
- get the correct Git branch,
- install project dependencies,
- launch the MLflow server,
- log parameters and models,
- load models from the registry by version or by tag.

## 1) Get the branch


```bash
git fetch upstream
git checkout 1-Starter-ML-Registry
git pull upstream 1-Starter-ML-Registry
```



## 2) Install dependencies

From the project root:

```bash
make setup
```

This runs:

```bash
pip install -e .
```

## 3) Launch MLflow

Start MLflow in a dedicated terminal:

```bash
make launch-mlflow-server
```

By default this project starts MLflow on:
- Tracking UI: `http://localhost:5000`
- Backend store: `sqlite:///mlflow.db`
- Artifact root: `./mlartifacts`

In another terminal, set the tracking URI before running scripts:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## 4) Basic example: log params, metrics, and a model

Run the notebook `notebooks/Experiement-Tracking.ipynb` to see how to log parameters, metrics, and models to MLflow.


## 5) Load a model from the registry

In the same notebook, see how to load a model from the registry using its name and version or using its name and stage (e.g. "production").

## 6) Next steps

- Try to implement the Mlflow logic inside the `diamonds` notebook.
- Implement Experiment Tracking in the training library `src/diamonds`:
  - log parameters (e.g. model hyperparameters),
  - log metrics (e.g. validation performance),
  - log the trained model to MLflow.
  - Load the model from MLflow and use it to make predictions.

Since we are on a regression what are some relevant metrics to log? 

<details>
<Summary> Solution:</Summary>
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
</details>
