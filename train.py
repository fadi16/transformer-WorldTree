import os.path

import numpy as np
import pandas as pd
from rich import box
from rich.table import Table, Column

from model_params import MAX_TARGET_TEXT_LENGTH, SEED, MAX_SOURCE_TEXT_LENGTH, MODEL, TRAIN_BATCH_SIZE, \
    VALID_BATCH_SIZE, TRAIN_EPOCHS, VAL_EPOCHS
import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from wt_dataset import WorldTreeDataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, Adafactor
from validate import validate
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from main_eval import evaluate, preprocess_predictions_df
import rich


def set_seed(model_params):
    torch.manual_seed(model_params[SEED])
    np.random.seed(model_params[SEED])
    torch.backends.cudnn.deterministic = True


def trainer(train_set: pd.DataFrame, dev_set: pd.DataFrame, source_text: str, target_text: str, model_params,
            output_dir="./outputs"):
    # for reproducibility
    set_seed(model_params)

    tb = SummaryWriter()

    print("LOADING MODEL ...\n")

    if "bart" in model_params[MODEL]:
        # BART tokenizer
        tokenizer = BartTokenizer.from_pretrained(model_params[MODEL])

        # BART model for conditional generation
        model = BartForConditionalGeneration.from_pretrained(model_params[MODEL])
        print("***************** TRAINING BART *******************")
    else:  # T5
        # t5-plain tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_params[MODEL])
        # t5-plain model for conditional generation
        model = T5ForConditionalGeneration.from_pretrained(model_params[MODEL])
        print("***************** TRAINING T5 *******************")

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

    if "bart" in model_params[model]:
        optimizer = AdamW(
            model.parameters(),
            lr=3e-5,
        )
    else:
        # optimizer: this is the optimizer recommended for t5-plain
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

    for training_epoch in range(model_params[TRAIN_EPOCHS]):
        print("STARTING TRAINING EPOCH: " + str(training_epoch) + "\n")
        loss = train_step(epoch=training_epoch,
                     tokenizer=tokenizer,
                     model=model,
                     device=device,
                     loader=training_loader,
                     optimizer=optimizer,
                     logger=training_logger)
        tb.add_scalar("Loss", loss, training_epoch)

        # evaluate at the end of each epoch
        print("Validating after training epoch #{0}\n".format(str(training_epoch)))
        for validation_epoch in range(model_params[VAL_EPOCHS]):
            predictions, actuals = validate(epoch=validation_epoch,
                                            tokenizer=tokenizer,
                                            loader=validation_loader,
                                            model=model,
                                            device=device,
                                            model_params=model_params)
            final_df = pd.DataFrame({
                "Questions": val_dataset[source_text],
                "Generated Text": predictions,
                "Actual Text": actuals
            })
            _, _, reference_text, _, _, _, generated_text_with_no_exact_repetitions, _, _, _ = preprocess_predictions_df(
                df=final_df)
            _, eval_score, _, _ = evaluate(metric_key="bleurt",
                                           generated=generated_text_with_no_exact_repetitions,
                                           references=reference_text,
                                           questions=val_dataset[source_text],
                                           best_and_worst=False)
            tb.add_scalar("Validation - bleurt score", eval_score, training_epoch)

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


def train_step(epoch, tokenizer, model, device, loader, optimizer, logger):
    model.train()

    # todo: data should be a batch of inputs
    for _, data in enumerate(loader, start=0):
        y = data["target_ids"].to(device, dtype=torch.long)
        # todo: what are these?
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        # In addition, we must make sure that padding token id’s of the labels are not taken into account by the loss function.
        # In PyTorch and Tensorflow, this can be done by replacing them with -100, which is the ignore_index of the CrossEntropyLoss
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        # todo: difference between lm_labels and y_ids
        outputs = model(
            input_ids=ids,  # ok
            attention_mask=mask,  # ok
            decoder_input_ids=y_ids,  # todo this is not needed according to the documentation
            labels=lm_labels
        )

        # FA: this is cross entropy loss between predicted and golden output
        loss = outputs[0]

        if _ % 100 == 0:
            logger.add_row(str(epoch), str(_), str(loss))
            rich.print(logger)

        # clears old gradients from last step - so that they do not accumulate everytime you do loss.backwards
        optimizer.zero_grad()
        # back propagations
        loss.backward()
        # gradient decent
        optimizer.step()
    return loss
