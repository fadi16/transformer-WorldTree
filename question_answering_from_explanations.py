import os
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

MODEL = "MODEL"
CHECKPOINT = "CHECKPOINT"
TRAIN_BATCH_SIZE = "TRAIN_BATCH_SIZE"
VALID_BATCH_SIZE = "VALID_BATCH_SIZE"
TRAIN_EPOCHS = "TRAIN_EPOCHS"
LEARNING_RATE = "LEARNING_RATE"
MAX_SOURCE_TEXT_LENGTH = "MAX_SOURCE_TEXT_LENGTH"
SEED = "SEED"
OUTPUT_DIR = "OUTPUT_DIR"

TRAIN_CSV_PATH_QA = "./data/v2-proper-data/train_qa.csv"
VAL_CSV_PATH_QA = "./data/v2-proper-data/dev_qa.csv"

MODEL_PARAMS = {
    MODEL: "BERT",
    CHECKPOINT: "bert-base-uncased",
    TRAIN_BATCH_SIZE: 16,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 10,
    LEARNING_RATE: 1e-5,
    MAX_SOURCE_TEXT_LENGTH: 512,
    SEED: 42,
    OUTPUT_DIR: "./output"
}


class QAWorldTreeDataset(Dataset):
    """
    We want the data to be:
    input to bert is:
        [CLS] question [SEP] explanation [SEP] candidate_answer, label
        question + contet ..., hypothesis 2, label

    """

    def __init__(self, dataframe, tokenizer, source_len):
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
        self.questions = self.data["question"]
        self.explanations = self.data["explanation"]
        self.candidate_answers = self.data["answer"]
        self.labels = self.data["label"]

        lengths = (np.array([len(q.split()) for q in self.questions])
                   + np.array([len(exp.split()) for exp in self.explanations])
                   + np.array([len(str(a).split()) for a in self.candidate_answers]) + 3
                   ).tolist()

        print("average length: ", np.mean(lengths))
        print("longest: ", max(lengths))

    def __len__(self):
        return len(self.labels) // 4 + 1

    def __getitem__(self, index):
        q_labels = []
        new_index = index * 4
        sources = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "labels": []}
        for i in range(4):
            # tokenizing source
            source = self.tokenizer(
                self.questions[new_index + i] + " [SEP] " + self.explanations[new_index + i], self.candidate_answers[new_index + i],
                pad_to_max_length=False,
                truncation=True,
                return_token_type_ids=True
            )

            q_labels.append(self.labels[new_index + i])

            sources["input_ids"].append(source["input_ids"])
            sources["attention_mask"].append(source["attention_mask"])
            sources["token_type_ids"].append(source["token_type_ids"])

        right_label = q_labels.index(max(q_labels))
        return {
            "input_ids": sources["input_ids"],
            "attention_mask": sources["attention_mask"],
            "token_type_ids": sources["token_type_ids"],
            "labels": right_label
        }


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def train_step(epoch, model, optimizer, training_loader, device, tb):
    model.train()
    train_losses = []
    for _, data in enumerate(training_loader, 0):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        labels = data['labels'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids, labels=labels)

        optimizer.zero_grad()

        loss = outputs.loss
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = np.mean(train_losses)
    tb.add_scalar("train_loss", average_train_loss, epoch)

    return average_train_loss


def val_step(epoch, model, val_loader, device, tb):
    model.eval()

    all_labels = []
    probs = []
    val_losses = []

    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids, labels=labels)

            loss = outputs.loss
            val_losses.append(loss.item())

            output_logits = outputs.logits

            all_labels.extend(labels.cpu().detach().numpy().tolist())
            probs.extend(torch.sigmoid(output_logits).cpu().detach().numpy().tolist())

    average_val_loss = np.mean(val_losses)
    val_accuracy = get_eval_scores(all_labels, probs)

    tb.add_scalar("val_loss", average_val_loss, epoch)
    tb.add_scalar("val_accuracy", val_accuracy, epoch)

    return average_val_loss, val_accuracy


def train_loop(params):
    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for producing graphs with tensorboard
    tb = SummaryWriter()

    train_df = pd.read_csv(TRAIN_CSV_PATH_QA)
    val_df = pd.read_csv(VAL_CSV_PATH_QA)

    tokenizer = AutoTokenizer.from_pretrained(params[CHECKPOINT])
    model = AutoModelForMultipleChoice.from_pretrained(params[CHECKPOINT]).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=params[LEARNING_RATE])

    train_dataset = QAWorldTreeDataset(train_df, tokenizer, params[MAX_SOURCE_TEXT_LENGTH])
    val_dataset = QAWorldTreeDataset(val_df, tokenizer, params[MAX_SOURCE_TEXT_LENGTH])

    collator = DataCollatorForMultipleChoice(tokenizer)

    # don't shuffle for the different candidate answers to appear next to each other
    # shuffling would mess us everything
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params[VALID_BATCH_SIZE],
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )

    print("begin training")
    accuracy, val_loss = val_step(-1, model, val_loader, device, tb)
    best_accuracy = accuracy
    print(f"initial validation loss = {val_loss}")
    for epoch in range(params[TRAIN_EPOCHS]):
        training_loss = train_step(epoch, model, optimizer, train_loader, device, tb)
        accuracy, val_loss = val_step(epoch, model, val_loader, device, tb)

        if best_accuracy < accuracy:
            best_accuracy = accuracy

            # save the best model so far
            model_checkpoint_path = os.path.join(params[OUTPUT_DIR], "checkpoints")
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print("SAVED MODEL AT " + model_checkpoint_path + "\n")

        print(f"Epoch {epoch} Done")
        print(f"Training Loss:\t{training_loss}")
        print(f"Validation Loss:\t{val_loss}")
        print(f"Current Accuracy:\t{accuracy}")
        print(f"Best Accuracy:\t{best_accuracy}")
        print("**" * 30)


def get_eval_scores(label_indicies, probs_arr):
    # labels = [[1 if i == label_index else 0 for i in range(4)] for label_index in label_indicies]
    # predicted_labels = [[1 if p == max(probs) else 0 for p in probs] for probs in probs_arr]

    predicted_labels_indicies = [probs.index(max(probs)) for probs in probs_arr]

    # no. correctly classified / total number of samples
    accuracy = metrics.accuracy_score(label_indicies, predicted_labels_indicies)

    return accuracy


def seed_for_reproducability(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    train_loop(MODEL_PARAMS)
