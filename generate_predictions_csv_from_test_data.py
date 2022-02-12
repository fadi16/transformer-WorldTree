import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer,BartForConditionalGeneration, BartTokenizer
import pandas as pd
from model_params import *
from wt_dataset import WorldTreeDataset
from torch.utils.data import DataLoader
from retrieve_prompt_generate import retrieve
###########################################
#TODO: WHY USE THE DIRECT CLASSES, CAN I USE THE GENERIC ONE SO THAT I WOULDN'T NEED A MODE
############################################
# todo: change checkpoint and file paths if needed
#############################################
MODE = "t5"
OUTPUT_FILE_PATH = "evaluation/validation_predictions_vs_actuals.csv"
MODEL_CHECKPOINT_DIR_PATH = "./outputs/checkpoints/T5-FromQnA-with-proper-data-splitting"
TEST_DATA_PATH = "./data/v2-proper-data/dev_data_wed.csv"
TRAIN_DATA_PATH = "./data/v2-proper-data/train_data_wed.csv"
target_text = "explanation"
##############################################

if __name__ == "__main__":
    if MODE == "bart":
        print("---- Using BART ----")
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        params_of_mode_model = bart_model_params
    else:
        print("---- Using T5 ----")
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        params_of_mode_model = t5_model_params


    df_test = pd.read_csv(TEST_DATA_PATH, delimiter="\t")
    df_train = pd.read_csv(TRAIN_DATA_PATH, delimiter="\t")

    if params_of_mode_model[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
        print("USING RETRIEVAL METHOD")
        train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                       testing_df=df_test,
                                                                       no_similar_hypotheses=t5_model_params[
                                                                           NO_SIMILAR_HYPOTHESIS],
                                                                       no_retrieved_facts=t5_model_params[
                                                                           NO_FACTS_TO_RETRIEVE])
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

    if MODE == "bart":
        model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
    else:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)

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
                #early_stopping=True # stop beam search once at least num_beams sentences are finished per batch
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

    predictions_and_actuals_df.to_csv(OUTPUT_FILE_PATH)