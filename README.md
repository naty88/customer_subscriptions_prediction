# User subscription prediction

## Repository Structure
- Notebooks:
  - EDA_analysis: [EDA_analysis.ipynb](notebooks%2FEDA_analysis.ipynb)
  - Model selection: [model_selection.ipynb](notebooks%2Fmodel_selection.ipynb)
- Source directory:
  - [src/get_predictions.py](src%2Fget_predictions.py)
  - [src/utils.py](src%2Futils.py)

## Usage / How to Run
- install dependencies from [requirements.txt](%2Frequirements.txt)
- select the best model and then train it with [model_selection.ipynb](notebooks%2Fmodel_selection.ipynb)
- start from src/ directory [get_predictions.py](src%2Fget_predictions.py) to generate predictions for [test_file.xlsx](data%2Ftest_file.xlsx)
    ```shell
    python get_predictions.py ../data/test_file.xlsx
    ```

- Hyperparameter tuning is an optional step, that you can run after training (also with [model_selection.ipynb](notebooks%2Fmodel_selection.ipynb))
  - model parameter will be optimized with [Optuna](https://optuna.readthedocs.io/en/stable/) framework
  - you can observe the optimization process starting optuna dashboard from `notebooks/` directory using CLI:
  ```shell
    optuna-dashboard sqlite:///db.sqlite3
    ```
  - TODO: get predisctions after tuning

## Dataset contains following features
 
TODO https://archive.ics.uci.edu/dataset/222/bank+marketing


| Feature name |    Type     | Description | Values|
|:-------------|:-----------:|:-----------:|:-----:|
|age           |   Integer   | 23.99       |       |
|job           | Categorical | 23.99       |       |
|marital       | Categorical | 19.99       |       |
|education     | Categorical | 42.99       |       |
|default       |    Binary   | 42.99       |       |
|housing       |    False    | 42.99       |       |
|loan          |    False    | 42.99       |       |
|contact       |    False    | 42.99       |       |
|month         |    False    | 42.99       |       |
|day_of_week   |    False    | 42.99       |       |
|duration      |    False    | 42.99       |       |
|campaign      |    False    | 42.99       |       |
|previous      |    False    | 42.99       |       |
|poutcome      |    False    | 42.99       |       |
|y             |    False    | 42.99       |       |

