import pandas as pd
from train import trainer
from model_params import *
from retrieve_prompt_generate import retrieve

if __name__ == "__main__":
    ####################### CHANGE AS APPROPRRIATE #######################
    path_train = "./data/v2-proper-data/train_data_wed.csv"
    path_dev = "./data/v2-proper-data/dev_data_wed.csv"
    chosen_model_params = bart_model_params
    ######################################################################

    df_train = pd.read_csv(path_train, delimiter="\t")
    print(df_train.head())

    df_dev = pd.read_csv(path_dev, delimiter="\t")
    print(df_dev.head())

    if chosen_model_params[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
        print("USING RETRIEVAL METHOD")
        train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                       testing_df=df_dev,
                                                                       no_similar_hypotheses=chosen_model_params[
                                                                           NO_SIMILAR_HYPOTHESIS],
                                                                       no_retrieved_facts=chosen_model_params[
                                                                           NO_FACTS_TO_RETRIEVE])
        for i in range(len(train_retrieved_facts)):
            df_train[chosen_model_params[TRAIN_ON]][i] += " @@ " + train_retrieved_facts[i]
        for i in range(len(dev_retrieved_facts)):
            df_dev[chosen_model_params[TRAIN_ON]][i] += " @@ " + dev_retrieved_facts[i]

    trainer(
        train_set=df_train,
        dev_set=df_dev,
        source_text=chosen_model_params[TRAIN_ON],
        target_text="explanation",
        model_params=chosen_model_params,
        output_dir="./outputs",
    )
