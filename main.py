import pandas as pd
from train_t5 import t5_trainer
from model_params import model_params

if __name__ == "__main__":
    path_train = "./data/eb_train_chains.csv"
    path_dev = "./data/eb_dev_chains.csv"

    df_train = pd.read_csv(path_train, delimiter="\t")
    print(df_train.head())

    df_dev = pd.read_csv(path_dev, delimiter="\t")
    print(df_dev.head())

    t5_trainer(
        train_set=df_train,
        dev_set=df_dev,
        source_text="Questions",
        target_text="Explanations",
        model_params=model_params,
        output_dir="./outputs",
    )