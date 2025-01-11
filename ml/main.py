import argparse
from scripts.train_hyperparameters_search import main_raytune_search_train
from scripts.parallel_prediction_ensamble import parallel_prediction_ensamble

def main():
    parser = argparse.ArgumentParser(description="Train model with hyperparameter search or run parallel prediction ensemble.")

    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "predict"),
        required=True,
        help="Mode of operation: train or predict.",
    )

    # Arguments related to data
    parser.add_argument(
        "--data_train_val_test",
        type=str,
        help="Path to the CSV file containing training, validation, and test data. Usually located in .data/processed folder.",
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

    # Arguments related to experiment results
    parser.add_argument(
        "--experiment_result_path",
        type=str,
        default="../../experiments",
        help="Path to the directory where experiment results will be saved.",
    )

    # Arguments specific to prediction mode
    parser.add_argument(
        "--model_experiments_path",
        type=str,
        help="Path to the directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--data_input_prediction",
        type=str,
        help="Path to the input data for prediction.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Metric to evaluate predictions (e.g., accuracy, precision, recall).",
    )
    parser.add_argument(
        "--result_path_prediction_csv",
        type=str,
        help="Result path of csv prediction",
    )
    parser.add_argument(
        "--top_n_hparams",
        type=int,
        default=40,
        help="Number of top hyperparameter configurations to use for ensemble predictions.",
    )
    parser.add_argument(
        "--n_checkpoints",
        type=int,
        default=None,
        help="Number of checkpoints to use for each model (default: all available).",
    )

    args = parser.parse_args()

    if args.mode == "train":
        if not args.data_train_val_test:
            raise ValueError("--data_train_val_test is required in train mode.")

        main_raytune_search_train(
            data_train_val_test=args.data_train_val_test,
            target_name_experiment=args.target_name_experiment,
            data_split_mode=args.data_split_mode,
            experiment_result_path=args.experiment_result_path,
        )

    elif args.mode == "predict":
        if not (args.model_experiments_path and args.data_input_prediction):
            raise ValueError("--model_experiments_path and --data_input are required in predict mode.")

        parallel_prediction_ensamble(
            model_experiments_path=args.model_experiments_path,
            data_input=args.data_input_prediction,
            metric=args.metric,
            result_path=args.result_path_prediction_csv,
            top_n_hparams=args.top_n_hparams,
            n_checkpoints=args.n_checkpoints,
        )

if __name__ == "__main__":
    main()