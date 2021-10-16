# MADE - Cassava-Leaf-Disease-Classification

==============================

## Identify the type of disease presesave Leaf image

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tests              <- tests to run and test model different functions


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Dataset
Download dataset from [Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification/data), indicate path in file entities/configs/configs.yaml.
Following paths should be set:
`
* data_path: "../../input/cassava-leaf-disease-classification/"
* train_csv_data_path: "../../input/cassava-leaf-disease-classification/train.csv"
* label_json_data_path: "../../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
* train_img_path: "../../input/cassava-leaf-disease-classification/train_images/"
* test_img_path: "../../input/cassava-leaf-disease-classification/test_images/"`

## Start
Change directory to cassava-leaf-disease-classification folder.
`cd cassava-leaf-disease-classification folder/`
Configs for modifications can be found in yaml entities/configs/configs.yaml.

## Enviroment
* Activate virtual environnment
* `pip install -r requirements.txt`

## EDA in jupyter notebook
```bash
jupyter notebook notebooks/cassava-leaf-disease-classification.ipynb
```

## Training

```bash
!python3 train_pipeline.py entities/configs/configs.yaml
```

## Prediction

```bash
????
```

## Local project testing
```bash
pytest -v . (not ready yet)
```

## Check code style with pylint
```bash
pylint cassava-leaf-disease-classification --disable=C0114,C0115,C0116 --fail-under=7.0 (not ready yet)
```
