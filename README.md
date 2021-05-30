epilepsy_regression_predictions
==============================

A short description of the project.

In this study we have used several Machine Learning models for the regression question of predicting number of Emergency visits. In an earlier study, a classification analysis was done to investigate a patient as a high utilizer if she visited the ED more than once in a year. With the clinical feature space comprising of diagnosis, ccs conditions, frailty indicator, disabled flags, palliative care indicators (over than 300 features) high accuracy was noted for the binary classification task. In this study the problem of regression has been undertaken. I use Ridge, Lasso, Elastic Net, Tree algorithms (CART, RandomForest, Adaboost) and Support Vector Machines. In epilepsy_nn.py a 3 layer Feed forward Neural Network model has been implemented. Mean squared error is reported. I have seperately imputed continuous (mean value) and categorical variables (most frequent). Categorical variables have been one hot encoded. (epilepsy_preprocess.py) I have used sklearn library in Python for the machine learning models. I have used both Randomized and Grid Search for cross-validation. I have implement a 5 fold cross-validation for hyper-parameters.

Setup:

Run conda env export -p ./env > environment.yml to create the required environment based on the required packages specified in environment.yml.

Run source env/bin/activate ./env or conda activate ./env to activate the created environment depending on which version of Conda you are using.

Code:

the code for running the model is in src/data. The code is supposed to be run in the sequence epilepsy_preprocess.py (for data pre-processing), epilepsy.py (for running the ML models) and if required deep learning, to run 

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
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
