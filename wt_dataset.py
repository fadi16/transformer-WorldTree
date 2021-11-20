import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd

pd.set_option('display.max_colwidth', 1000)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from model_params import *
from rich.table import Column, Table
from rich import box
import rich


def display_df(df):
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )
    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    rich.print(table)


class WorldTreeDataset(Dataset):
    """
    creating dataset to be passed to the dataloader and then to the neural network
    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text_column_name, target_text_column_name):
        """
        :param dataframe: pandas.DataFrame, the input data frame
        :param tokenizer: transformers.tokenizer
        :param source_len: maximum length of source
        :param target_len: maximum length of target
        :param source_text: column name of source text
        :param target_text: column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_text_column_name]
        self.target_text = self.data[target_text_column_name]

    # todo: ??
    def __len__(self):
        """returns the length of the dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return input ids, attention marks and target ids"""
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # clean data, make sure it's a string
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        # tokenizing source
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            # todo what about target_mask ??
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


if __name__ == "__main__":
    training_data_path = "./data/eb_train_chains.csv"
    dev_data_path = "./data/eb_dev_chains.csv"

    df_train = pd.read_csv(training_data_path, delimiter="\t")
    df_dev = pd.read_csv(dev_data_path, delimiter="\t")

    display_df(df_dev[["Questions", "Explanations"]].head(1))
    # print(df_dev.head(1)["Explanations"])
    #
    # tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    #
    # testing_set = WorldTreeDataset(
    #     dataframe=df_dev,
    #     tokenizer=tokenizer,
    #     source_text_column_name="Questions",
    #     target_text_column_name="Explanations",
    #     source_len=model_params["MAX_SOURCE_TEXT_LENGTH"],
    #     target_len=model_params["MAX_TARGET_TEXT_LENGTH"]
    # )
    #
    # print(testing_set[10])
    # for i in range(10):
    #    print(testing_set[i])
