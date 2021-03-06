import os.path
import pandas as pd
from rich import box
from rich.table import Table, Column

from model_params import *
import torch
import numpy as np
from generate import generate, generate_with_chains, generate_with_inference_chains
from torch.utils.tensorboard import SummaryWriter
import rich
from generate_v2_data import CENTRAL, GROUNDING, LEXGLUE
from postprocess import postprocess_explanation
from eval_metrics import evaluate_bleurt


def set_seed(model_params):
    torch.manual_seed(model_params[SEED])
    np.random.seed(model_params[SEED])
    torch.backends.cudnn.deterministic = True


def trainer(model, tokenizer, optimizer, training_loader, validation_loader, validation_loader2, chosen_model_params,
            output_dir="./outputs"):
    # for reproducibility
    set_seed(chosen_model_params)

    tb = SummaryWriter()

    # send to GPU/TPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("USING DEVICE " + device)
    model = model.to(device)

    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    best_val_score = -1
    best_val_loss = 1000

    for training_epoch in range(chosen_model_params[TRAIN_EPOCHS]):
        print("STARTING TRAINING EPOCH: " + str(training_epoch) + "\n")

        training_loss = train_step(epoch=training_epoch,
                                   tokenizer=tokenizer,
                                   model=model,
                                   device=device,
                                   loader=training_loader,
                                   optimizer=optimizer,
                                   logger=training_logger)
        tb.add_scalar("train_loss", training_loss, training_epoch)

        print("training_loss = ", training_loss)

        # evaluate at the end of each epoch
        print("Validating after training epoch #{0}\n".format(str(training_epoch)))
        for validation_epoch in range(chosen_model_params[VAL_EPOCHS]):

            val_loss = val_step(training_epoch, tokenizer, model, device, validation_loader)
            tb.add_scalar("val_loss", val_loss, training_epoch)

            if chosen_model_params[CHAIN]:
                if chosen_model_params[CHAIN_ON] == ROLE:
                    # overall bleurt scores
                    (questions, retrieved_central, retrieved_grounding,
                     retrieved_lexglue, predictions, actuals) = generate_with_chains(epoch=validation_epoch,
                                                                                     tokenizer=tokenizer,
                                                                                     loader=validation_loader2,
                                                                                     model=model,
                                                                                     device=device,
                                                                                     model_params=chosen_model_params)
                    # augment retrieved to questions
                    if retrieved_lexglue and retrieved_central and retrieved_grounding:
                        for i in range(len(questions)):
                            questions[i] += " @@ " + " || ".join(
                                [retrieved_central[i], retrieved_grounding[i], retrieved_lexglue[i]])

                    # todo
                    reference_text = [postprocess_explanation(exp) for exp in actuals]
                    generated_text = [postprocess_explanation(exp) for exp in predictions]

                    bleurt_scores = evaluate_bleurt(reference_text, generated_text)
                    eval_score = np.mean(bleurt_scores)

                    final_df = pd.DataFrame({
                        "Questions": questions,
                        "Generated Text": generated_text,
                        "Actual Text": reference_text,
                        "bleurt_scores": bleurt_scores
                    })

                    #######################################################
                    # bleurt scores for each explanatory role

                    questions_chains, predictions_chains, actuals_chains = generate(epoch=validation_epoch,
                                                                                    tokenizer=tokenizer,
                                                                                    loader=validation_loader,
                                                                                    model=model,
                                                                                    device=device,
                                                                                    chosen_model_params=chosen_model_params)

                    roles = [CENTRAL, GROUNDING, LEXGLUE] if chosen_model_params[CENTRAL_FIRST] else [GROUNDING,
                                                                                                      CENTRAL, LEXGLUE]

                    reference_text = [postprocess_explanation(exp) for exp in actuals_chains]
                    generated_text = [postprocess_explanation(exp) for exp in predictions_chains]

                    for role_index, role in enumerate(roles):
                        role_ref = [reference_text[i] for i in range(len(reference_text)) if i % 3 == role_index]
                        role_gen = [generated_text[i] for i in range(len(generated_text)) if i % 3 == role_index]
                        role_questions = [questions_chains[i] for i in range(len(questions_chains)) if
                                          i % 3 == role_index]

                        role_bleurt_scores = evaluate_bleurt(role_ref, role_gen)
                        role_eval_score = np.mean(bleurt_scores)

                        print("{0} bleurt score = {1}".format(role, role_eval_score))
                        tb.add_scalar("{0}_BLEURT".format(role), role_eval_score)

                        role_df = pd.DataFrame({
                            "Questions": role_questions,
                            "Generated Text": role_gen,
                            "Actual Text": role_ref,
                            "bleurt_scores": role_bleurt_scores
                        })
                        role_df.to_csv("{0}_predictions_{1}.csv".format(role, training_epoch))

                # elif chosen_model_params[CHAIN_ON] == PREVIOUS_SORTED:
                #     questions, predictions, actuals = generate_with_inference_chains(epoch=validation_epoch,
                #                                                                      tokenizer=tokenizer,
                #                                                                      loader=validation_loader2,
                #                                                                      model=model,
                #                                                                      device=device,
                #                                                                      model_params=chosen_model_params)
                #     final_df = pd.DataFrame({
                #         "Questions": questions,
                #         "Generated Text": predictions,
                #         "Actual Text": actuals
                #     })
                #     _, _, reference_text, _, _, _, generated_text_with_no_exact_repetitions, _, _, _ = preprocess_predictions_df(
                #         df=final_df)
                #     _, eval_score, _, _ = evaluate_bleurt(metric_key="bleurt",
                #                                           generated=generated_text_with_no_exact_repetitions,
                #                                           references=reference_text,
                #                                           questions=None,
                #                                           best_and_worst=False)
                #     print("**" * 10)
                #     print("Finished overall validation")
                #     print("**" * 10)
                #
                #     #######################################
                #     # For each inference step
                #     questions_inference_steps, predictions_inference_steps, actuals_inference_steps = generate(
                #         epoch=validation_epoch,
                #         tokenizer=tokenizer,
                #         loader=validation_loader,
                #         model=model,
                #         device=device,
                #         chosen_model_params=chosen_model_params)
                #
                #     df = pd.DataFrame({
                #         "Questions": questions_inference_steps,
                #         "Generated Text": predictions_inference_steps,
                #         "Actual Text": actuals_inference_steps
                #     })
                #     _, _, reference_text, _, _, _, generated_text_with_no_exact_repetitions, _, _, _ = preprocess_predictions_df(
                #         df=df)
                #
                #     for inference_step in range(chosen_model_params[NO_INFERENCE_STEPS] + 1):
                #         inference_step_ref = [reference_text[i] for i in range(len(reference_text)) if
                #                               i % (chosen_model_params[NO_INFERENCE_STEPS] + 1) == inference_step]
                #         inference_step_gen = [generated_text_with_no_exact_repetitions[i] for i in
                #                               range(len(generated_text_with_no_exact_repetitions)) if
                #                               i % (chosen_model_params[NO_INFERENCE_STEPS] + 1) == inference_step]
                #         inference_step_questions = [questions_inference_steps[i] for i in
                #                                     range(len(questions_inference_steps)) if
                #                                     i % (chosen_model_params[NO_INFERENCE_STEPS] + 1) == inference_step]
                #
                #         _, inference_step_eval_score, _, _ = evaluate_bleurt(metric_key="bleurt",
                #                                                              generated=inference_step_gen,
                #                                                              references=inference_step_ref,
                #                                                              questions=None,
                #                                                              best_and_worst=False)
                #
                #         print("inference step {0} bleurt score = {1}".format(inference_step, inference_step_eval_score))
                #         tb.add_scalar("inference_step_{0}_BLEURT".format(inference_step), inference_step_eval_score)
                #
                #         inference_step_df = pd.DataFrame({
                #             "Questions": inference_step_questions,
                #             "Generated Text": inference_step_gen,
                #             "Actual Text": inference_step_ref
                #         })
                #         inference_step_df.to_csv("inference_step_{0}_predictions.csv".format(inference_step))
                #         print("**" * 10)
                #         print("finished inference step {0} validation".format(inference_step))
                #         print("**" * 10)


            else:
                questions, predictions, actuals = generate(epoch=validation_epoch,
                                                           tokenizer=tokenizer,
                                                           loader=validation_loader,
                                                           model=model,
                                                           device=device,
                                                           chosen_model_params=chosen_model_params)

                reference_text = [postprocess_explanation(exp) for exp in actuals]
                generated_text = [postprocess_explanation(exp) for exp in predictions]

                bleurt_scores = evaluate_bleurt(reference_text, generated_text)
                eval_score = np.mean(bleurt_scores)

                final_df = pd.DataFrame({
                    "Questions": questions,
                    "Generated Text": generated_text,
                    "Actual Text": reference_text,
                    "bleurt_scores": bleurt_scores
                })

            print("overall bleurt score = ", eval_score)
            tb.add_scalar("overall_bleurt_score", eval_score, training_epoch)

            if eval_score > best_val_score:
                best_val_score = eval_score

            print("best_bleurt_score = ", best_val_score)

            final_df.to_csv("predictions_{0}.csv".format(str(training_epoch)))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save predictions
                final_df.to_csv(os.path.join(output_dir, "best_predictions.csv"))
                print("SAVED BEST PREDICTIONS (based on val loss) AT " + os.path.join(output_dir,
                                                                                      "predictions.csv") + "\n")
                # save model and tokenizer
                model_checkpoint_path = os.path.join(output_dir, "checkpoints")
                model.save_pretrained(model_checkpoint_path)
                tokenizer.save_pretrained(model_checkpoint_path)
                print("SAVED MODEL AT " + model_checkpoint_path + "\n")

            print(f"validation_loss = {val_loss}")
            print(f"best_validation_loss = {best_val_loss}")
            print("**" * 20)


def metric_agnostic_trainer(model, tokenizer, optimizer, training_loader, validation_loader, validation_loader2,
                            chosen_model_params,
                            output_dir="./outputs"):
    # for reproducibility
    set_seed(chosen_model_params)

    tb = SummaryWriter()

    # send to GPU/TPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("USING DEVICE " + device)
    model = model.to(device)

    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    best_validation_loss = 1000

    for training_epoch in range(chosen_model_params[TRAIN_EPOCHS]):
        print("STARTING TRAINING EPOCH: " + str(training_epoch) + "\n")
        training_loss = train_step(training_epoch, tokenizer, model, device, training_loader, optimizer,
                                   training_logger)
        print(f"training_loss = {training_loss}")
        tb.add_scalar("training_loss", training_loss, training_epoch)

        # evaluate at the end of each epoch
        print("Validating after training epoch #{0}\n".format(str(training_epoch)))
        for validation_epoch in range(chosen_model_params[VAL_EPOCHS]):
            if chosen_model_params[CHAIN]:
                if chosen_model_params[CHAIN_ON] == ROLE:
                    (questions, retrieved_central, retrieved_grounding,
                     retrieved_lexglue, predictions, actuals) = generate_with_chains(epoch=validation_epoch,
                                                                                     tokenizer=tokenizer,
                                                                                     loader=validation_loader2,
                                                                                     model=model,
                                                                                     device=device,
                                                                                     model_params=chosen_model_params,
                                                                                     no_samples=None)
                    print("retrieved_central = ", retrieved_central)
                    print("retrieved_grounding = ", retrieved_grounding)
                    print("retrieved_lexglue = ", retrieved_lexglue)

            else:
                questions, predictions, actuals = generate(epoch=validation_epoch,
                                                           tokenizer=tokenizer,
                                                           loader=validation_loader,
                                                           model=model,
                                                           device=device,
                                                           chosen_model_params=chosen_model_params, no_samples=None)
            print("--" * 20)

            validation_loss = val_step(training_epoch, tokenizer, model, device, validation_loader)
            tb.add_scalar("validation_loss", validation_loss, training_epoch)
            print(f"validation_loss = {validation_loss}")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                print(f"best_validation_loss = {best_validation_loss}")
                # save model and tokenizer
                model_checkpoint_path = os.path.join(output_dir, "checkpoints")
                model.save_pretrained(model_checkpoint_path)
                tokenizer.save_pretrained(model_checkpoint_path)
                print("SAVED MODEL AT " + model_checkpoint_path + "\n")

            print("VALIDATION DONE")

        print("**" * 20)


# This function is copied from modeling_bart.py
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def train_step(epoch, tokenizer, model, device, loader, optimizer, logger):
    model.train()

    training_losses = []
    for _, data in enumerate(loader, start=0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = shift_tokens_right(y, tokenizer.pad_token_id)
        lm_labels = y[:, :].clone().detach()
        # In addition, we must make sure that padding token id???s of the labels are not taken into account by the loss
        # function. In PyTorch and Tensorflow, this can be done by replacing them with -100, which is the
        # ignore_index of the CrossEntropyLoss
        lm_labels[y[:, :] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,  # ok
            attention_mask=mask,  # ok
            decoder_input_ids=y_ids,
            labels=lm_labels
        )

        loss = outputs[0]

        # clears old gradients from last step - so that they do not accumulate everytime you do loss.backwards
        optimizer.zero_grad()
        # back propagations
        loss.backward()
        # gradient decent
        optimizer.step()

        training_losses.append(loss.item())

    return np.mean(training_losses)


def val_step(epoch, tokenizer, model, device, loader):
    model.eval()

    val_losses = []
    with torch.no_grad():
        for _, data in enumerate(loader, start=0):
            y = data["target_ids"].to(device, dtype=torch.long)
            y_ids = shift_tokens_right(y, tokenizer.pad_token_id)
            lm_labels = y[:, :].clone().detach()
            lm_labels[y[:, :] == tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels
            )
            loss = outputs[0]
            val_losses.append(loss.item())

    return np.mean(val_losses)
