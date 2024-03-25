## TrypanoDEEPScreen: A Deep Learning Model for Compound Activity Prediction with Hyperparameter Tuning

This repository provides a comprehensive deep learning pipeline for training and evaluating a model to predict compound activity, based on [ahmetrifaioglu's DEEPScreen](https://github.com/cansyl/DEEPScreen), publicated in [Chemical Science on 2020](https://doi.org/10.1039/C9SC03414E). The model, TrypanoDEEPScreen, is implemented using PyTorch Lightning and leverages Ray Tune for efficient hyperparameter search.


### Code Structure

The code is organized into the following directories:

* `src`: Contains the main source code for the project.
    * `ml`: Code for training, evaluating, and predicting with the model.
        * `data`: Code for data loading, preprocessing, and spliting.
        * `datasets`: Code for defining custom datasets.
        * `engine`: Code for defining the deep learning model architecture and training loop.
        * `scripts`: Scripts for training, hyperparameter tuning, and prediction.
        * `utils`: Utility functions for logging, configurations, and exceptions.

#### Understanding the Codebase

The codebase is organized into several key modules:

- **`main.py`**: This script serves as the entry point for the application
- **`train_hyperparameters_search.py`**: This script handles hyperparameter search for DeepScreen using Ray Tune.
- **`engine/`**: This directory contains the core components for model definition and training/testing logic.
    - **`system.py`**: Defines the TrypanoDEEPScreen model architecture as a subclass of `LightningModule` from PyTorch 
- **`datasets/`**: This directory contains code for data loading and preprocessing.
  - **`datamodule.py`**: Defines a custom PyTorch Lightning `DataModule` for handling data loading and splitting.
  - **`dataset.py`**: Defines pytorch dataloaders.

### Data Preparation

The code expects a CSV file containing the following columns:

* `comp_id`: Unique identifier for the compound
* `smiles`: SMILES string representing the compound structure
* `bioactivity`: Binary label indicating the activity of the compound (0: inactive, 1: active)

**Data Splitting:**

The code currently supports non-random splitting of the data into training, validation, and test sets. The data frame should have an additional column named `"data_split"` that specifies the split for each data point (e.g., `"train"`, `"validation"`, or `"test"`).

**Preprocessing:**

The code includes functionalities for data preprocessing, such as handling missing values and converting SMILES strings into image representations.

### Training and Hyperparameter Tuning

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

This will train the model, perform hyperparameter search, and save the results in the specified experiment directory.

### Testing the Model

After training, you can use the trained model to predict the activity of new compounds. The script `src/ml/scripts/predict.py` can be used for making predictions. This script requires the following arguments:

* `--model_path`: Path to the checkpoint file containing the trained model.
* `--data_path`: Path to the CSV file containing the compounds for prediction.
* `--experiment_result_path`: Path to the experiment directory where the predictions will be saved.




## Usage Guide:

**1. Installation:**

**Conda Environment Installation Guide for DeepScreen**

This guide details the steps to install the necessary dependencies for the DeepScreen project using a conda environment. It assumes you have conda installed on your system. If not, you can download it from [https://docs.conda.io/projects/conda/en/4.14.x/dev-guide/deep-dive-install.html](https://docs.conda.io/projects/conda/en/4.14.x/dev-guide/deep-dive-install.html).

**1. Locate the `trypanodeepscreen_conda_env_che.yml` file:**

   - Navigate to the directory containing the `trypanodeepscreen_conda_env_che.yml` file. This file likely resides in the `src/ml` directory of your DeepScreen project based on the provided path (`/home/sjinich/disco/TrypanoDEEPscreen/src/ml/trypanodeepscreen_conda_env_che.yml`).

**2. Create the conda environment:**

   - Open a terminal window and navigate to the directory containing the `trypanodeepscreen_conda_env_che.yml` file using the `cd` command.
   - Run the following command, replacing `<environment_name>` with the desired name for your conda environment (e.g., `deepscreen_env`):

     ```bash
     conda env create -f trypanodeepscreen_conda_env_che.yml -n <environment_name>
     ```

   - This command will create a new conda environment named `<environment_name>` and install the packages specified in the `trypanodeepscreen_conda_env_che.yml` file.

**3. Activate the environment (optional but recommended):**

   - Once the environment is created, you can activate it to isolate the installed packages from other conda environments you might have.
   - To activate the environment, run:

     ```bash
     conda activate <environment_name>
     ```

   - Now, when you run DeepScreen commands, it will use the packages installed in the activated environment named `<environment_name>`.

**Verification:**

   - To verify that the environment is created and activated correctly, you can check the currently active environment using:

     ```bash
     conda env list
     ```

   - The output should list your environment (`<environment_name>`) with an asterisk (*) next to it, indicating that it's active.

**Additional Considerations:**

- If you encounter any errors during the installation process, double-check the path to the `trypanodeepscreen_conda_env_che.yml` file and ensure you have a stable internet connection.
- The `trypanodeepscreen_conda_env_che.yml` file might have specific channel configurations or custom package versions. Make sure your conda configuration allows for installation from those channels.
- For more advanced conda environment management, refer to the conda documentation: [https://conda.io/projects/conda/en/latest/commands/install.html](https://conda.io/projects/conda/en/latest/commands/install.html)


**2. Running the Code:**

**a) Training and Hyperparameter Tuning:**

To train DeepScreen with hyperparameter search, run:

```bash
python main.py --data_train_val_test <path/to/data.csv>
                   --target_name_experiment <experiment_name>
                   --data_split_mode <split_mode>
                   --experiment_result_path <path/to/results>
```

**Arguments:**

- `--data_train_val_test`: Path to the CSV file containing compound activity data with separate columns for training, validation, and test sets.
- `--target_name_experiment`: Name assigned to this experiment for logging and identification purposes.
- `--data_split_mode`: Mode for splitting the data into training, validation, and test sets (e.g., 'non_random_split'). Refer to the `datasets/` code for available options.
- `--experiment_result_path`: Path to the directory where experiment