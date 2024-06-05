# Customer subscription prediction

## Project Overview

This project aims to predict customer subscriptions for term deposits using a dataset from a bank marketing campaign.
The project involves exploratory data analysis (EDA), model selection, training, and hyperparameter tuning to optimize
the prediction model's performance.

## Repository Structure

- Notebooks:
    - EDA analysis: [EDA_analysis.ipynb](notebooks%2FEDA_analysis.ipynb)
    - Model selection: [model_selection.ipynb](notebooks%2Fmodel_selection.ipynb)
- Source directory:
    - Prediction Script: [src/get_predictions.py](src%2Fget_predictions.py)
    - Utility functions: [src/utils.py](src%2Futils.py)

## Installation Instructions

To set up the project environment, clone the repository and install the required dependencies
from [requirements.txt](%2Frequirements.txt) file:

```shell
pip install -r requirements.txt
```

## Usage guidelines

### Model Selection and Training

1. Perform exploratory data analysis and model selection using the Jupyter notebook:
   ```shell
   jupyter lab notebooks/model_selection.ipynb
   ```
2. Train the selected model and perform optional hyperparameter tuning using the same notebook.

### Hyperparameter Tuning

1. Hyperparameter tuning can be performed after training using [Optuna](https://optuna.readthedocs.io/en/stable/)
   framework.
2. Monitor the optimization process with the Optuna Dashboard. Start the dashboard from the _root_
   directory `user_subscription_prediction/` using CLI:
    3. Run:
   ```shell
   optuna-dashboard sqlite:///db.sqlite3
   ```
    4. Go to: http://127.0.0.1:8080/

### Generating Predictions

1. To generate predictions for a given test file, use the [src/get_predictions.py](src%2Fget_predictions.py) script. Run
   the script form `src/` directory:
   ```shell
   python src/get_predictions.py data/test_file.xlsx
   ```
2. If you have performed hyperparameter tuning, you can use the tuned model by adding the `-t True` flag:
   ```shell
   python src/get_predictions.py data/test_file.xlsx -t True
   ```

## The dataset includes the following features

| Feature name |    Type     |                              Description                              |           Example of categorical features           |
|:-------------|:-----------:|:---------------------------------------------------------------------:|:---------------------------------------------------:|
| age          |   integer   |                                 23.99                                 |                                                     |
| job          | categorical |                              type of job                              |       'admin', 'blue-collar', 'entrepreneur'        |
| marital      | categorical |                            marital status                             |           'divorced', 'married', 'single'           |
| education    | categorical |                            education level                            | 'basic.4y', 'basic.6y', 'high.school', 'illiterate' |
| default      |   binary    |                     has client credit in default?                     |                                                     |
| housing      |   binary    |                       has clienet housing loan?                       |                                                     |
| loan         |   binary    |                       has client personal loan?                       |                                                     |
| contact      | categorical |                      contact communication type                       |               'cellular', 'telephone'               |
| month        | categorical |                      last contact month of year                       |                 'jan', 'feb', 'mar'                 |
| day_of_week  | categorical |                     last contact day of the week                      |                 'sun', 'mon', 'tue'                 |
| duration     |   integer   |                   last contact duration, in seconds                   |                                                     |
| campaign     |   integer   | number of contacts performed during this campaign and for this client |                                                     |
| previous     |   integer   | number of contacts performed before this campaign and for this client |                                                     |
| poutcome     | categorical |              outcome of the previous marketing campaign               |         'failure', 'nonexistent', 'success'         |
| y            |   binary    |               has the client subscribed a term deposit?               |                                                     |

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing