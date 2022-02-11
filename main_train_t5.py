import pandas as pd
from train_t5 import t5_trainer
from model_params import *
from retrieve_prompt_generate import retrieve

if __name__ == "__main__":
    path_train = "./data/v2-proper-data/train_data_wed.csv"
    path_dev = "./data/v2-proper-data/dev_data_wed.csv"

    df_train = pd.read_csv(path_train, delimiter="\t")
    print(df_train.head())

    df_dev = pd.read_csv(path_dev, delimiter="\t")
    print(df_dev.head())

    if t5_model_params[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
        print("USING RETRIEVAL METHOD")
        train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                       testing_df=df_dev,
                                                                       no_similar_hypotheses=t5_model_params[
                                                                           NO_SIMILAR_HYPOTHESIS],
                                                                       no_retrieved_facts=t5_model_params[
                                                                           NO_FACTS_TO_RETRIEVE])
        for i in range(len(train_retrieved_facts)):
            df_train[t5_model_params[TRAIN_ON]][i] += " @@ " + train_retrieved_facts[i]
        for i in range(len(dev_retrieved_facts)):
            df_dev[t5_model_params[TRAIN_ON]][i] += " @@ " + dev_retrieved_facts[i]

    t5_trainer(
        train_set=df_train,
        dev_set=df_dev,
        source_text=t5_model_params[TRAIN_ON],
        target_text="explanation",
        model_params=t5_model_params,
        output_dir="./outputs",
    )
