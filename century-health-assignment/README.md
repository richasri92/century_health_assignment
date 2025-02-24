# century_health_assignment

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.11`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

ADDING STEPS TO EXECUTE / FIND THE RESULT AS PER THE TASKS LISTED IN THE ASSIGNMENT:


## 1. Data Assessment:
    1. Draw a diagram to give a sense of how the different datasets link together ---> FIND ATTACHED DIAGRAM UNDER
    2. Examine each dataset and write down all data quality issues you find. -->Please refer [data_inspection.ipynb (https://github.com/richasri92/century_health_assignment/blob/main/century-health-assignment/notebooks/data_inspection.ipynb) notebook under noteboks folder


## 2. Data Pipelining:

You can run the Kedro project with:

```
kedro run --pipeline=data_engineering
```


## 3. Data Testing: 

The test case is written under `src/tests/test_split_symptoms.py` Run the tests as follows:

```
pytest src/tests/test_split_symptoms.py
```

## 4. Data Analysis::

Execute a run-all on the notebook listed as data_analysis.ipynb under notebooks folder


