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

### The dataset includes the following features
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
| previous     |   integer   | number of contacts performed before this campaign and for this client |        'failure', 'nonexistent', 'success')         |
| poutcome     | categorical |              outcome of the previous marketing campaign               |                                                     |
| y            |   binary    |              has the client subscribed a term deposit?                |                                                     |

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing