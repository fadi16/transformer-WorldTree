import pandas as pd
from train_t5 import t5_trainer
from model_params import model_params

if __name__ == "__main__":
    path_train = "./data/v2-proper-data/train_data.csv"
    path_dev = "./data/v2-proper-data/dev_data.csv"


    df_train = pd.read_csv(path_train, delimiter="\t")
    print(df_train.head())

    df_dev = pd.read_csv(path_dev, delimiter="\t")
    print(df_dev.head())

    t5_trainer(
        train_set=df_train,
        dev_set=df_dev,
        source_text="hypothesis",
        target_text="explanation",
        model_params=model_params,
        output_dir="./outputs",
    )