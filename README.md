# PitchSim: Precision Pitch Prediction
## Overview
PitchSim is an advanced pitch analysis framework that leverages machine learning, clustering, and statistical modeling to simulate baseball pitches and calculate their expected run values and event rates. By isolating the intrinsic characteristics of a pitch (i.e. "stuff"),  PitchSim provides a focused understanding of pitch performance, independent of usage and command.

### Key Features:

* **Fuzzy Clustering:** Categorizes pitches into clusters based on stuff features.
* **Location Distributions:** Models pitch locations using Kernel Density Estimation (KDE).
* **Directed Acyclic Graph (DAG) Modeling:** Organizes pitch events in a DAG structure.
* **Gradient Boosted Decision Trees (XGBoost):** Utilizes XGBoost for predictive modeling.
* **Bias Correction:** Addresses selection bias by fixing usage and command for each cluster
* **Model Distillation:** Offers efficient simulation through model approximation.

### Presentation
[Saberseminar 2023 Slides](https://docs.google.com/presentation/d/1LtN8Vc_66ec7wGFujLar18oyE8EFrZFUpTanj_UYAFI/edit?usp=sharing)

## Installation
1. Clone the repository:
`git clone https://github.com/caldred/PitchSim.git`
2. Navigate to the project directory:
`cd PitchSim`
3. Install the required dependencies, either with:
`conda env create --file environment.yml`
or:
`pip install -r requirements.txt`

## Usage 
The `run.py` script runs all the steps necessary to build PitchSim. It is currently not configured with command line arguments, so make sure to open the file and and edit if needed. You can run the script with simply `python run.py`, but given how long some of the steps take, you may find it useful to execute each step separately in `run.ipynb`

## Documentation
Detailed documentation is available in the respective source files.

* **`data_loader.py`:** Functions for loading and handling data.
* **`processing.py`:** Data preprocessing and feature engineering.
* **`stuff_model.py`:** Modeling and clustering of pitch stuff.
* **`tune_xgboost.py`:** Hyperparameter tuning for XGBoost models.
* **`run.py`:** Main script to load the data, train the models, and run the simulation.
* **`run.ipynb`:** Jupyter notebook verison of `run.py`

## Results
Full results up to August 2023 are available [here](https://docs.google.com/spreadsheets/d/1019d4XW4BvVidaFuzcXY7wdwGWEdjcTf4BnNqYUoaMM/edit?usp=sharing)

## Contact
Find the author on Twitter: [@CalAldred](https://twitter.com/CalAldred)

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details






