import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
import pandas as pd
from model_params import *
from wt_dataset import WorldTreeDataset, GROUNDING_RETRIEVED, CENTRAL_RETRIEVED, LEXGLUE_RETRIEVED
from torch.utils.data import DataLoader
from retrieve_prompt_generate import retrieve
from generate import get_chain_source_ids_and_source_mask
from generate_v2_data import explanatory_role_to_sep, GROUNDING, BACKGROUND, CENTRAL, LEXGLUE
from main_eval import CENTRAL_FACTS_SEP, GROUNDING_FACTS_SEP, LEXGLUE_FACTS_SEP
from generate import generate, generate_with_chains

# todo: change checkpoint and file paths if needed
#############################################
OUTPUT_FILE_PATH = "evaluation/validation_predictions_vs_actuals.csv"
MODEL_CHECKPOINT_DIR_PATH = "./outputs/checkpoints"
target_text = "explanation"
TRAINING_CSV_PATH = "./data/v2-proper-data/train_data_wed.csv"
TESTING_CSV_PATH = "./data/v2-proper-data/dev_data_wed.csv"
chosen_model_params = bart_chain_model_params
##############################################

if __name__ == "__main__":
    if "bart" in chosen_model_params[MODEL]:
        print("---- Using BART ----")
        tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)

    else:
        print("---- Using T5 ----")
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_CHECKPOINT_DIR_PATH)

    df_test = pd.read_csv(TESTING_CSV_PATH, delimiter="\t")
    df_train = pd.read_csv(TRAINING_CSV_PATH, delimiter="\t")

    if chosen_model_params[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
        if chosen_model_params[CHAIN]:
            print("USING RETRIEVAL METHOD - chain")
            central_train_retrieved_facts, central_dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                                           testing_df=df_test,
                                                                                           no_similar_hypotheses=
                                                                                           chosen_model_params[
                                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                                           no_retrieved_facts=
                                                                                           chosen_model_params[
                                                                                               NO_FACTS_TO_RETRIEVE],
                                                                                           only_central=True,
                                                                                           retrieved_facts_sep=CENTRAL_FACTS_SEP)
            print("finished retrieving central facts")
            grounding_train_retrieved_facts, grounding_dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                                               testing_df=df_test,
                                                                                               no_similar_hypotheses=
                                                                                               chosen_model_params[
                                                                                                   NO_SIMILAR_HYPOTHESIS],
                                                                                               no_retrieved_facts=
                                                                                               chosen_model_params[
                                                                                                   NO_FACTS_TO_RETRIEVE],
                                                                                               only_grounding=True,
                                                                                               retrieved_facts_sep=GROUNDING_FACTS_SEP)
            print("finished retrieving grounding facts")
            lexglue_train_retrieved_facts, lexglue_dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                                           testing_df=df_test,
                                                                                           no_similar_hypotheses=
                                                                                           chosen_model_params[
                                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                                           no_retrieved_facts=
                                                                                           chosen_model_params[
                                                                                               NO_FACTS_TO_RETRIEVE],
                                                                                           only_lexglue=True,
                                                                                           retrieved_facts_sep=LEXGLUE_FACTS_SEP)
            testing_dataset = WorldTreeDataset(
                dataframe=df_test,
                tokenizer=tokenizer,
                target_len=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
                source_len=chosen_model_params[MAX_SOURCE_TEXT_LENGTH],
                target_text_column_name=target_text,
                source_text_column_name=chosen_model_params[TRAIN_ON],
                central_retrieved=central_dev_retrieved_facts if chosen_model_params[
                    AUGMENT_INPUT_WITH_RETRIEVED_FACTS] else [],
                grounding_retrieved=grounding_dev_retrieved_facts if chosen_model_params[
                    AUGMENT_INPUT_WITH_RETRIEVED_FACTS] else [],
                lexglue_retrieved=lexglue_dev_retrieved_facts if chosen_model_params[
                    AUGMENT_INPUT_WITH_RETRIEVED_FACTS] else []
            )
        else:
            print("USING RETRIEVAL METHOD - no chain")
            train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                           testing_df=df_test,
                                                                           no_similar_hypotheses=chosen_model_params[
                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                           no_retrieved_facts=chosen_model_params[
                                                                               NO_FACTS_TO_RETRIEVE],
                                                                           only_central=chosen_model_params[ONLY_CETRAL])
            for i in range(len(train_retrieved_facts)):
                df_train[chosen_model_params[TRAIN_ON]][i] += " @@ " + train_retrieved_facts[i]
            for i in range(len(dev_retrieved_facts)):
                df_test[chosen_model_params[TRAIN_ON]][i] += " @@ " + dev_retrieved_facts[i]

            testing_dataset = WorldTreeDataset(
                dataframe=df_test[[chosen_model_params[TRAIN_ON], target_text]],
                tokenizer=tokenizer,
                target_len=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
                source_len=chosen_model_params[MAX_SOURCE_TEXT_LENGTH],
                target_text_column_name=target_text,
                source_text_column_name=chosen_model_params[TRAIN_ON]
    )

    testing_loader = DataLoader(
        dataset=testing_dataset,
        batch_size=4,  # todo doesn't make a lot of sense why this works in validate
        shuffle=False,
        num_workers=0
    )

    predictions = []
    actuals = []
    retrieved_central_facts = []
    retrieved_grounding_facts = []
    retrieved_lexglue_facts = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if chosen_model_params[CHAIN]:
            (questions, all_retrieved_central_facts, all_retrieved_grounding_facts,
             all_retrieved_lexglue_facts, predictions, actuals) = generate_with_chains(0, tokenizer, model, device, testing_loader, chosen_model_params)

            predictions_and_actuals_df = pd.DataFrame({
                "Questions": df_test[chosen_model_params[TRAIN_ON]],
                "Generated Text": predictions,
                "Actual Text": actuals,
                CENTRAL_RETRIEVED: retrieved_central_facts,
                GROUNDING_RETRIEVED: retrieved_grounding_facts,
                LEXGLUE_RETRIEVED: retrieved_lexglue_facts
            })
        else:
            questions, predictions, actuals = generate(0, tokenizer, model, device, testing_loader, chosen_model_params)

            predictions_and_actuals_df = pd.DataFrame({
                "Questions": df_test[chosen_model_params[TRAIN_ON]],
                "Generated Text": predictions,
                "Actual Text": actuals
            })

    predictions_and_actuals_df.to_csv(OUTPUT_FILE_PATH)