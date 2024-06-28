import argparse

from scripts.train_hyperparameters_search_rdkit_features import main_raytune_search_train


def main():
    parser = argparse.ArgumentParser(description="Train model with hyperparameter search using Ray Tune.")

    # Arguments related to data
    parser.add_argument(
        "--data_train_val_test",
        type=str,
        help="Path to the CSV file containing training, validation, and test data. Ussualy located in .data/processed folder",
    )
    parser.add_argument(
        "--target_name_experiment",
        type=str,
        default="deepscreen_experiment",
        help="Name of the experiment to be used for logging and identification.",
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="non_random_split",
        help="Mode for splitting data into training, validation, and test sets.",
    )

    # Argument related to experiment results
    parser.add_argument(
        "--experiment_result_path",
        type=str,
        default="../../.experiments",
        help="Path to the directory where experiment results will be saved.",
    )

    args = parser.parse_args()

    main_raytune_search_train(
        data_train_val_test=args.data_train_val_test,
        target_name_experiment=args.target_name_experiment,
        data_split_mode=args.data_split_mode,
        experiment_result_path=args.experiment_result_path,
    )


if __name__ == "__main__":
    main()