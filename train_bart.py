import os.path

import numpy as np
import pandas as pd
from rich import box
from rich.table import Table, Column

from model_params import MAX_TARGET_TEXT_LENGTH, SEED, MAX_SOURCE_TEXT_LENGTH, MODEL, TRAIN_BATCH_SIZE, \
    VALID_BATCH_SIZE, TRAIN_EPOCHS, VAL_EPOCHS, LEARNING_RATE
import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from wt_dataset import WorldTreeDataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, Adafactor
from datasets import load_metric
from train import train
from validate import validate


def set_seed(model_params):
    torch.manual_seed(model_params[SEED])
    np.random.seed(model_params[SEED])
    torch.backends.cudnn.deterministic = True


def bart_trainer(train_set: pd.DataFrame, dev_set: pd.DataFrame, source_text: str, target_text: str, model_params,
               output_dir="./outputs"):
    # for reproducibility
    set_seed(model_params)

    print("LOADING MODEL ...\n")

    # BART tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_params[MODEL])

    # BART model for conditional generation
    model = BartForConditionalGeneration.from_pretrained(model_params[MODEL])

    # send to GPU/TPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("USING DEVICE " + device)
    model = model.to(device)

    # importing data
    train_dataset = train_set[[source_text, target_text]]
    print(f"TRAIN Dataset: {train_dataset.shape}\n")

    training_dataset = WorldTreeDataset(
        dataframe=train_dataset,
        tokenizer=tokenizer,
        target_len=model_params[MAX_TARGET_TEXT_LENGTH],
        source_len=model_params[MAX_SOURCE_TEXT_LENGTH],
        target_text_column_name=target_text,
        source_text_column_name=source_text
    )
    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=model_params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0
    )

    val_dataset = dev_set[[source_text, target_text]]
    print(f"VALIDATION Dataset: {val_dataset.shape}\n")

    validation_dataset = WorldTreeDataset(
        dataframe=val_dataset,
        tokenizer=tokenizer,
        target_len=model_params[MAX_TARGET_TEXT_LENGTH],
        source_len=model_params[MAX_SOURCE_TEXT_LENGTH],
        target_text_column_name=target_text,
        source_text_column_name=source_text
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=model_params[VALID_BATCH_SIZE],
        shuffle=False,
        num_workers=0
    )

    # todo: what about test set?

    # optimizer = AdamW(
    #     params=model.parameters(),
    #     lr=model_params[LEARNING_RATE]
    # )

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    best_val_score = -1
    # loading the evaluation metric: with key of metric from huggingface an the metric's configuration
    metric = load_metric("bleurt", "bleurt-large-512")

    for training_epoch in range(model_params[TRAIN_EPOCHS]):
        print("STARTING TRAINING EPOCH: " + str(training_epoch) + "\n")
        train(epoch=training_epoch,
              tokenizer=tokenizer,
              model=model,
              device=device,
              loader=training_loader,
              optimizer=optimizer,
              logger=training_logger)

        # evaluate at the end of each epoch
        print("Validating after training epoch #{0}\n".format(str(training_epoch)))
        for validation_epoch in range(model_params[VAL_EPOCHS]):
            predictions, actuals = validate(epoch=validation_epoch,
                                            tokenizer=tokenizer,
                                            loader=validation_loader,
                                            model=model,
                                            device=device,
                                            model_params=model_params)
            print("Predictions vs Actuals")
            for i in range(len(predictions)):
                print("Prediction = ",predictions[i])
                print("Actual = ", actuals[i])

            metric.add_batch(predictions=predictions,
                             references=actuals)
            eval_score = np.mean(metric.compute()["scores"])
            final_df = pd.DataFrame({
                "Questions": val_dataset[source_text],
                "Generated Text": predictions,
                "Actual Text": actuals
            })

            if eval_score > best_val_score:
                best_val_score = eval_score
                # save predictions
                final_df.to_csv(os.path.join(output_dir, "predictions.csv"))
                print("SAVED PREDICTIONS AT " + os.path.join(output_dir, "predictions.csv") + "\n")
                # save model and tokenizer
                # todo why save tokenizer?
                model_checkpoint_path = os.path.join(output_dir, "checkpoints")
                model.save_pretrained(model_checkpoint_path)
                tokenizer.save_pretrained(model_checkpoint_path)
                print("SAVED MODEL AT " + model_checkpoint_path + "\n")

            print("VALIDATION DONE - BEST BLEURT SCORE = {0}, CURRENT BLEURT SCORE = {1}\n".format(best_val_score,
                                                                                                   eval_score))
