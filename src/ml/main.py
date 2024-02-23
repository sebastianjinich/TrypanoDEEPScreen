from scripts.train_hyperparameters_search import main_raytune_search_train

def main():
    main_raytune_search_train(data_train_val_test="/home/sjinich/disco/TrypanoDEEPscreen/.data/processed/CHEMBL5567.csv",
                              target_name_experiment="CHEMBL5567",
                              data_split_mode="non_random_split",
                              experiment_result_path="../../.experiments")
    
if __name__ == "__main__":
    main()