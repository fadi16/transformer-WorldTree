import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
import pandas as pd
from model_params import *
from wt_dataset import WorldTreeDataset, GROUNDING_RETRIEVED, CENTRAL_RETRIEVED, LEXGLUE_RETRIEVED
from torch.utils.data import DataLoader
from retrieve_prompt_generate import retrieve
from main_eval import CENTRAL_FACTS_SEP, GROUNDING_FACTS_SEP, LEXGLUE_FACTS_SEP
from generate import generate, generate_with_chains, generate_with_inference_chains
from postprocess import postprocess_explanation
from eval_metrics import *
from generation_params import *
import pickle
# todo: change checkpoint and file paths if needed
#############################################
OUTPUT_FILE_PATH = "test.csv"
MODEL_CHECKPOINT_DIR_PATH = "./evaluation/bart-plain-metric-agnostic/checkpoint"
target_text = "explanation"
TRAINING_CSV_PATH = "./data/v2-proper-data/train_data_wed.csv"
TESTING_CSV_PATH = "./data/v2-proper-data/dev_data_wed.csv"
chosen_model_params = bart_plain_model_params
do_eval = True
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
                                                                                               NO_FACTS_TO_RETRIEVE_CENTRAL],
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
                                                                                                   NO_FACTS_TO_RETRIEVE_GROUNDING],
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
                                                                                               NO_FACTS_TO_RETRIEVE_LEXGLUE],
                                                                                           only_lexglue=True,
                                                                                           retrieved_facts_sep=LEXGLUE_FACTS_SEP)
        else:
            print("USING RETRIEVAL METHOD - no chain")
            train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                           testing_df=df_test,
                                                                           no_similar_hypotheses=chosen_model_params[
                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                           no_retrieved_facts=chosen_model_params[
                                                                               NO_FACTS_TO_RETRIEVE],
                                                                           only_central=chosen_model_params[
                                                                               ONLY_CETRAL])
            for i in range(len(train_retrieved_facts)):
                df_train[chosen_model_params[TRAIN_ON]][i] += " @@ " + train_retrieved_facts[i]
            for i in range(len(dev_retrieved_facts)):
                df_test[chosen_model_params[TRAIN_ON]][i] += " @@ " + dev_retrieved_facts[i]

    testing_dataset = WorldTreeDataset(
        dataframe=df_test,
        tokenizer=tokenizer,
        target_len=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
        source_len=chosen_model_params[MAX_SOURCE_TEXT_LENGTH],
        target_text_column_name=target_text,
        source_text_column_name=chosen_model_params[TRAIN_ON],
        central_retrieved=central_dev_retrieved_facts if chosen_model_params[
                                                             AUGMENT_INPUT_WITH_RETRIEVED_FACTS] and
                                                         chosen_model_params[CHAIN] else [],
        grounding_retrieved=grounding_dev_retrieved_facts if chosen_model_params[
                                                                 AUGMENT_INPUT_WITH_RETRIEVED_FACTS] and
                                                             chosen_model_params[CHAIN] else [],
        lexglue_retrieved=lexglue_dev_retrieved_facts if chosen_model_params[
                                                             AUGMENT_INPUT_WITH_RETRIEVED_FACTS] and
                                                         chosen_model_params[CHAIN] else []
    )

    testing_loader = DataLoader(
        dataset=testing_dataset,
        batch_size=8,
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

    grid_search_gen_params = get_grid_search_params()

    best_bleurt_score = -1
    best_config = None

    postprocessed_actuals = None

    scores = []
    configs = []

    for num, gen_params in enumerate(grid_search_gen_params):

        print(f"Generation Params : config{num} out of {len(grid_search_gen_params)}")
        for k, v in gen_params.items():
            print(f"{k}:\t{v}")

        if chosen_model_params[CHAIN]:
            if chosen_model_params[CHAIN_ON] == ROLE:
                (questions, retrieved_central_facts, retrieved_grounding_facts,
                 retrieved_lexglue_facts, predictions, actuals) = generate_with_chains(0, tokenizer, model, device,
                                                                                       testing_loader,
                                                                                       chosen_model_params,
                                                                                       gen_params=gen_params,
                                                                                       verbose=False)

                predictions_and_actuals_df = pd.DataFrame({
                    "Questions": df_test[chosen_model_params[TRAIN_ON]],
                    "Generated Text": predictions,
                    "Actual Text": actuals,
                    CENTRAL_RETRIEVED: retrieved_central_facts if retrieved_central_facts else ([" "] * len(actuals)),
                    GROUNDING_RETRIEVED: retrieved_grounding_facts if retrieved_grounding_facts else (
                                [" "] * len(actuals)),
                    LEXGLUE_RETRIEVED: retrieved_lexglue_facts if retrieved_lexglue_facts else ([" "] * len(actuals))
                })

                print(f"question: {questions[0]}")
                if retrieved_central_facts:
                    print(f"retrieved_central: {retrieved_central_facts[0]}")
                if retrieved_grounding_facts:
                    print(f"retrieved_grounding: {retrieved_grounding_facts[0]}")
                if retrieved_lexglue_facts:
                    print(f"retrieved_lexglue: {retrieved_lexglue_facts[0]}")
                print(f"actual: {actuals[0]}")
                print(f"predicted = {predictions[0]}")

            elif chosen_model_params[CHAIN_ON] == PREVIOUS_SORTED:
                questions, predictions, actuals = generate_with_inference_chains(0, tokenizer, model,
                                                                                 device, testing_loader,
                                                                                 chosen_model_params,
                                                                                 gen_params, verbose=False)
                predictions_and_actuals_df = pd.DataFrame({
                    "Questions": questions,
                    "Generated Text": predictions,
                    "Actual Text": actuals
                })
        else:
            questions, predictions, actuals = generate(0, tokenizer, model, device, testing_loader, chosen_model_params,
                                                       gen_params=gen_params, verbose=False)

            predictions_and_actuals_df = pd.DataFrame({
                "Questions": df_test[chosen_model_params[TRAIN_ON]],
                "Generated Text": predictions,
                "Actual Text": actuals
            })

        if do_eval:

            if not postprocessed_actuals:
                postprocessed_actuals = [postprocess_explanation(actual_exp) for actual_exp in actuals]
            postprocessed_generated = [postprocess_explanation(gen_exp) for gen_exp in predictions]

            # bleurt
            bleurt_scores = evaluate_bleurt(postprocessed_actuals, postprocessed_generated)
            mean_bleurt = np.mean(bleurt_scores)

            scores.append(mean_bleurt)
            configs.append(gen_params)

            predictions_and_actuals_df["bleurt_score"] = bleurt_scores
            print(f"mean_bleurt =\t{mean_bleurt}")

            if mean_bleurt > best_bleurt_score:
                best_bleurt_score = mean_bleurt
                best_config = gen_params
                print(f"best_bleurt_score = {best_bleurt_score}")

            predictions_and_actuals_df.to_csv(f"{gen_params[NAME]}.csv")
            print("**" * 20)

    print("Best Configuration:")
    print(best_config)
    print("Best Score:")
    print(best_bleurt_score)

    f = open("configs_and_scores.pkl", "wb")
    pickle.dump(zip(configs, scores), f)


            # bleu4_scores = evaluate_bleu(postprocessed_actuals, postprocessed_generated, "bleu4")
            # mean_bleu4 = np.mean(bleu4_scores)
            #
            # predictions_and_actuals_df["bleu4_score"] = bleu4_scores
            # print(f"mean_bleu4 =\t{mean_bleu4}")
            #
            # # rouge-l-sum
            # rouge_l_sum_p_scores, rouge_l_sum_r_scores, rouge_l_sum_f1_scores = evaluate_rouge(postprocessed_actuals,
            #                                                                                    postprocessed_generated,
            #                                                                                    "rougeLsum")
            #
            # mean_rouge_l_sum_p = np.mean(rouge_l_sum_p_scores)
            # mean_rouge_l_sum_r = np.mean(rouge_l_sum_r_scores)
            # mean_rouge_l_sum_f1 = np.mean(rouge_l_sum_f1_scores)
            #
            # predictions_and_actuals_df["rouge_l_sum_p_score"] = rouge_l_sum_p_scores
            # predictions_and_actuals_df["rouge_l_sum_r_score"] = rouge_l_sum_r_scores
            # predictions_and_actuals_df["rouge_l_sum_f1_score"] = rouge_l_sum_f1_scores
            #
            # print(f"mean_rouge_l_sum_p =\t{mean_rouge_l_sum_p}")
            # print(f"mean_rouge_l_sum_r =\t{mean_rouge_l_sum_r}")
            # print(f"mean_rouge_l_sum_f1 =\t{mean_rouge_l_sum_f1}")
            #
            # # rouge-4
            # rouge4_p_scores, rouge4_r_scores, rouge4_f1_scores = evaluate_rouge(postprocessed_actuals,
            #                                                                     postprocessed_generated, "rouge4")
            # mean_rouge4_p = np.mean(rouge4_p_scores)
            # mean_rouge4_r = np.mean(rouge4_r_scores)
            # mean_rouge4_f1 = np.mean(rouge4_f1_scores)
            #
            # predictions_and_actuals_df["rouge_4_score_p"] = rouge4_p_scores
            # predictions_and_actuals_df["rouge_4_score_r"] = rouge4_r_scores
            # predictions_and_actuals_df["rouge_4_score_f1"] = rouge4_f1_scores
            #
            # print(f"mean_rouge4_p =\t{mean_rouge4_p}")
            # print(f"mean_rouge4_r =\t{mean_rouge4_r}")
            # print(f"mean_rouge4_f1 =\t{mean_rouge4_f1}")
            #
            # # relevance, completeness, binary completeness, f-measure
            # question_ids = df_test["question_id"]
            # relevance_scores, completeness_scores, binary_completeness_scores, f_measure_relevance_completeness_scores = evaluate_relevance_and_completeness(
            #     question_ids, postprocessed_generated)
            #
            # mean_relevance = np.mean(relevance_scores)
            # mean_completeness = np.mean(completeness_scores)
            # mean_binary_completeness = np.mean(binary_completeness_scores)
            # mean_relevance_completeness_f_measure = np.mean(f_measure_relevance_completeness_scores)
            #
            # print(f"mean_relevance =\t{mean_relevance}")
            # print(f"mean_completeness =\t{mean_completeness}")
            # print(f"mean_binary_completeness =\t{mean_binary_completeness}")
            # print(f"mean_relevance_completeness_f_measure =\t{mean_relevance_completeness_f_measure}")
            #
            # predictions_and_actuals_df["relevance_score"] = relevance_scores
            # predictions_and_actuals_df["completeness_score"] = completeness_scores
            # predictions_and_actuals_df["binary_completeness_score"] = binary_completeness_scores
            # predictions_and_actuals_df["f_measure_relevance_completeness_score"] = f_measure_relevance_completeness_scores
