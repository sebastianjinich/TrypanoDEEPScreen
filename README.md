# TrypanoDEEPScreen: Convolutional neural networks for drug-target interaction prediction based on 2D molecule structures

This repository provides a comprehensive deep learning pipeline for training and evaluating a model to predict compound activity, based on [ahmetrifaioglu's DEEPScreen](https://github.com/cansyl/DEEPScreen), publication in [Chemical Science on 2020](https://doi.org/10.1039/C9SC03414E). TrypanoDEEPScreen is implemented using PyTorch Lightning and leverages Ray Tune for efficient hyperparameter search. GPU usage and parallelization (when available) are completely handled by the pipeline.

## Understanding the Codebase

The code is organized into the following directories:

* `src`: Contains the main source code for the project.
    * `ml`: Code for training, evaluating, and predicting with the model.
        * `data`: Code for creating the datasets.
        * `datasets`: Code for defining custom datasets.
          - **`datamodule.py`**: Defines a custom PyTorch Lightning `DataModule` for handling data loading and splitting.
          - **`dataset.py`**: Defines pytorch dataloaders.  
        * `engine`: Code for defining the deep learning model architecture and training loop.
          - **`system.py`**: Defines the TrypanoDEEPScreen model architecture as a subclass of `LightningModule` from PyTorch
          - **`hyperparameters_tune_raytune.py`**: Defines the class responsible for hyperparameter search using Ray Tune. It automates the exploration of different model configurations, saving you valuable time and resources. 
        * `scripts`: Scripts for training, hyperparameter tuning, and prediction.
          - **`train_hyperparameters_search.py`**: This script streamlines the hyperparameter search and model training using Ray Tune. It calls upon the hyperparameters_tune_raytune.py script for the heavy lifting.
        * `utils`: Utility functions for logging, configurations, and exceptions.
          - **`configurations.py`** manages configurations.
      - **`main.py`**: This script serves as the entry point for the application
      - **`trypanodeepscreen_conda_env_che.yml`**: This file provides a recipe for creating a conda environment using conda, a popular package manager for scientific computing. This ensures you have all the necessary dependencies installed to run the code.

## **Installation**

### **Conda Environment Installation Guide for TrypanoDEEPScreen**

This guide details the steps to install the necessary dependencies for the TrypanoDEEPScreen project using a conda environment. It assumes you have conda installed on your system. If not, you can download it from [https://docs.conda.io/projects/conda/en/4.14.x/dev-guide/deep-dive-install.html](https://docs.conda.io/projects/conda/en/4.14.x/dev-guide/deep-dive-install.html).

**1. Locate the `trypanodeepscreen_conda_env_che.yml` file:**

   - Navigate to the directory containing the `trypanodeepscreen_conda_env_che.yml` file. This file likely resides in the `src/ml` directory of your TrypanoDEEPScreen project based on the provided path.

     ```bash
       cd TrypanoDEEPscreen/src/ml
     ```

**2. Create the conda environment:**

   - Run the following command:

     ```bash
     conda env create -f trypanodeepscreen_conda_env_che.yml
     ```

     This command will create a new conda environment named `TrypanoDEEPScreen_env` and install the packages specified in the `trypanodeepscreen_conda_env_che.yml` file.

**3. Activate the environment:**

   - Once the environment is created, you can activate it to isolate the installed packages from other conda environments you might have.
   - To activate the environment, run:

     ```bash
     conda activate TrypanoDEEPScreen_env
     ```

   - Now, when you run TrypanoDEEPScreen commands, it will use the packages installed in the activated environment named `TrypanoDEEPScreen_env`.

## Data Preparation

The code expects a CSV file containing the following columns:

* `comp_id`: Unique identifier for the compound
* `smiles`: SMILES string representing the compound structure
* `bioactivity`: Binary label indicating the activity of the compound (0: inactive, 1: active)
* `data_split`: Specifies the split for each data point (e.g., `"train"`, `"validation"`, or `"test"`). Only necessary for non-radom splitting.

### **Dataset generation and splitting**

Pipelines for creating datasets from ChEMBL database can be found in `data/`.

The code currently supports non-random splitting of the data into training, validation, and test sets. This means that the user must provide the data already splitted and specifies the split for each data point in the `data_split` column (e.g., `"train"`, `"validation"`, or `"test"`). 

### **Preprocessing**

The code includes functionalities for data preprocessing, such as handling missing values and converting SMILES strings into image representations.

## Training and Hyperparameter Tuning

The main script for training the model and performing hyperparameter search is `src/ml/main.py`. You can run this script with the following arguments:

* `--data_train_val_test`: Path to the CSV file containing the training, validation, and test data.
* `--target_name_experiment`: Name of the experiment for logging and identification.
* `--data_split_mode`: Mode for splitting data into training, validation, and test sets (currently supports `"non_random_split"`).
* `--experiment_result_path`: Path to the directory where experiment results will be saved (default: `../../.experiments`).

The script uses Ray Tune to perform a hyperparameter search to find the best model configuration. The search space and other hyperparameter tuning settings are defined in `src/ml/utils/configurations.py`.

### Running the Script

```bash
python src/ml/main.py \
--data_train_val_test path/to/your/data.csv \
--target_name_experiment my_experiment \
--data_split_mode non_random_split
```

This will train the model, perform a hyperparameter search, and save the results in the specified experiment directory.

### Using the trained model

**COMING SOON... (Still in development)**