import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer,BartForConditionalGeneration, BartTokenizer
import pandas as pd
from model_params import t5_model_params, MAX_SOURCE_TEXT_LENGTH, MAX_TARGET_TEXT_LENGTH
from wt_dataset import WorldTreeDataset
from torch.utils.data import DataLoader
###########################################
#TODO: WHY USE THE DIRECT CLASSES, CAN I USE THE GENERIC ONE SO THAT I WOULDN'T NEED A MODE
############################################
# todo: change checkpoint and file paths if needed
#############################################
MODE = "t5"
OUTPUT_FILE_PATH = "evaluation/T5/validation_predictions_vs_actuals-T5-from-QnA-with-data-splitting.csv"
MODEL_CHECKPOINT_DIR_PATH = "./outputs/checkpoints/T5-FromQnA-with-proper-data-splitting"
TEST_DATA_PATH = "./data/v2-proper-data/dev_data.csv"
source_text = "question_and_answer"
target_text = "explanation"
##############################################

if __name__ == "__main__":
    if MODE == "bart":
        print("---- Using BART ----")
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
    else:
        print("---- Using T5 ----")
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)

    df_test = pd.read_csv(TEST_DATA_PATH, delimiter="\t")

    testing_dataset = WorldTreeDataset(
        dataframe=df_test[[source_text, target_text]],
        tokenizer=tokenizer,
        target_len=t5_model_params[MAX_TARGET_TEXT_LENGTH],
        source_len=t5_model_params[MAX_SOURCE_TEXT_LENGTH],
        target_text_column_name=target_text,
        source_text_column_name=source_text
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


            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=t5_model_params[MAX_TARGET_TEXT_LENGTH],
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
        "Questions": df_test[source_text],
        "Generated Text": predictions,
        "Actual Text": actuals
    })

    predictions_and_actuals_df.to_csv(OUTPUT_FILE_PATH)