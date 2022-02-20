import pickle

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer,BartForConditionalGeneration, BartTokenizer
import pandas as pd
from model_params import *
from wt_dataset import WorldTreeDataset
from torch.utils.data import DataLoader
from retrieve_prompt_generate import retrieve
from main_eval import evaluate, preprocess_predictions_df
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
###########################################
#TODO: WHY USE THE DIRECT CLASSES, CAN I USE THE GENERIC ONE SO THAT I WOULDN'T NEED A MODE
############################################
# todo: change checkpoint and file paths if needed
#############################################
MODEL_CHECKPOINT_DIR_PATH = "../outputs/checkpoints/"
DEV_DATA_PATH = "../data/v2-proper-data/dev_data_wed.csv"
TRAIN_DATA_PATH = "../data/v2-proper-data/train_data_wed.csv"
target_text = "explanation"
params_of_mode_model = bart_model_params
MAX_NO_HYPOTHESES = 5
MAX_NO_FACTS = 8
##############################################


def plot_no_hypotheses_no_facts_vs_score():
    # todo: add x, y labels
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    f = open("./evaluation/BART-retrieve-prompt-central-only/retrieval_stretegies_scores_bart_central_only.pkl", "rb")
    data = pickle.load(f)

    no_hypotheses = np.array([h_num for h_num,_,_ in data])
    no_facts = np.array([f_num for _,f_num,_ in data])
    scores = np.array([s for _,_,s in data])

    ax.plot_trisurf(no_hypotheses, no_facts, scores,
                cmap='viridis', edgecolor='none')
    ax.set_title('surface');
    plt.show()

if __name__ == "__main__":
    no_hypotheses_no_facts_score = []
    best_score = -1
    best_no_hypotheses = -1
    best_no_facts = -1

    if "bart" in params_of_mode_model[MODEL]:
        print("---- Using BART ----")
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        params_of_mode_model = bart_model_params
        mode = "bart" + ("_central_only" if params_of_mode_model[ONLY_CETRAL] else "")
    else:
        print("---- Using T5 ----")
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        params_of_mode_model = t5_model_params
        mode = "t5" + ("_central_only" if params_of_mode_model[ONLY_CETRAL] else "")

    for no_hypotheses in range(1, MAX_NO_HYPOTHESES+1):
        for no_facts in range(1, MAX_NO_FACTS+1):
            df_test = pd.read_csv(DEV_DATA_PATH, delimiter="\t")
            df_train = pd.read_csv(TRAIN_DATA_PATH, delimiter="\t")

            if params_of_mode_model[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
                print("USING RETRIEVAL METHOD: ", "no_facts = ", no_facts, ", no_hypotheses =", no_hypotheses)
                train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                               testing_df=df_test,
                                                                               no_similar_hypotheses=no_hypotheses,
                                                                               no_retrieved_facts=no_facts,
                                                                               only_central=params_of_mode_model[ONLY_CETRAL])
                for i in range(len(train_retrieved_facts)):
                    df_train[params_of_mode_model[TRAIN_ON]][i] += " @@ " + train_retrieved_facts[i]
                for i in range(len(dev_retrieved_facts)):
                    df_test[params_of_mode_model[TRAIN_ON]][i] += " @@ " + dev_retrieved_facts[i]


            testing_dataset = WorldTreeDataset(
                dataframe=df_test[[params_of_mode_model[TRAIN_ON], target_text]],
                tokenizer=tokenizer,
                target_len=params_of_mode_model[MAX_TARGET_TEXT_LENGTH],
                source_len=params_of_mode_model[MAX_SOURCE_TEXT_LENGTH],
                target_text_column_name=target_text,
                source_text_column_name=params_of_mode_model[TRAIN_ON]
            )

            testing_loader = DataLoader(
                dataset=testing_dataset,
                batch_size=4, # todo doesn't make a lot of sense why this works in validate
                shuffle=False,
                num_workers=0
            )

            predictions = []
            actuals = []

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(testing_loader, start=0):
                    source_ids = data["source_ids"].to(device, dtype=torch.long)
                    source_mask = data["source_mask"].to(device, dtype=torch.long)
                    target_ids = data["target_ids"].to(device, dtype=torch.long)

                    # better than:
                    # this top p top k, with 0.95 (50/100)
                    generated_ids = model.generate(
                        input_ids=source_ids,
                        attention_mask=source_mask,
                        max_length=params_of_mode_model[MAX_TARGET_TEXT_LENGTH],
                        num_beams=2,
                        repetition_penalty=2.5, # todo: theta 1.2 with greedy reported to have worked well based on paper
                        length_penalty=1.0, # todo: this greater than 1 encourages model to generate longer sentences and vice versa
                        early_stopping=True # stop beam search once at least num_beams sentences are finished per batch
                    )

                    predicted_explanations = [tokenizer.decode(generated_id, skip_special_tokens=True, cleanup_tokenization_spaces=True)
                                              for generated_id in generated_ids]
                    actual_explanations = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in
                                           target_ids]

                    predictions.extend(predicted_explanations)
                    actuals.extend(actual_explanations)

            predictions_and_actuals_df = pd.DataFrame({
                "Questions": df_test[params_of_mode_model[TRAIN_ON]],
                "Generated Text": predictions,
                "Actual Text": actuals
            })

            _, _, reference_text, _, _, _, generated_text_with_no_exact_repetitions, _, _, _ = preprocess_predictions_df(
                df=predictions_and_actuals_df)
            _, eval_score, _, _ = evaluate(metric_key="bleurt",
                                           generated=generated_text_with_no_exact_repetitions,
                                           references=reference_text,
                                           questions=df_test[params_of_mode_model[TRAIN_ON]],
                                           best_and_worst=False)
            if eval_score > best_score:
                best_score = eval_score
                best_no_facts = no_facts
                best_no_hypotheses = no_hypotheses
                print("best score so far = ", best_score, ", best_no_facts = ", best_no_facts, ", best_no_hypotheses = ", best_no_hypotheses)

            no_hypotheses_no_facts_score.append((no_hypotheses, no_facts, eval_score))

    with open("retrieval_stretegies_scores_{0}.pkl".format(mode), "wb") as f:
        pickle.dump(no_hypotheses_no_facts_score, f)
