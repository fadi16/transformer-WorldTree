import json

import numpy as np
import pandas as pd
from retrieve_prompt_generate import retrieve

from wtv2_constants import *
from main_eval import *

# TRAINING_DATA_JSON_PATH = "./data/v2-proper-data/train_set_shared.json"
DEV_TATA_JSON_PATH = "./data/v2-proper-data/dev_set_shared.json"

DEV_CSV_DATA_PATH = "./data/v2-proper-data/dev_data_wed.csv"
TRAINING_CSV_DATA_PATH = "./data/v2-proper-data/train_data_wed.csv"


# todo low EP scores

def retrieve_facts_based_on_explanatory_power(bm25_model, training_df, testing_df, no_similar_hypotheses,
                                              no_retrieved_facts,
                                              fit_on="hypothesis"):
    ## for all roles - general
    _, retrieved_ids_to_scores_all_roles = retrieve.retrieve(training_df=training_df, testing_df=testing_df,
                                                             no_similar_hypotheses=no_similar_hypotheses,
                                                             no_retrieved_facts=no_retrieved_facts,
                                                             bm25_model=bm25_model, fit_on=fit_on,
                                                             ret_ids_to_scores=True, based_on_ep=True,
                                                             retrieve_training=False)
    retrieved_all_roles = [[fact_id for fact_id in retrieved_ids_to_scores_all_roles_for_q] for
                           retrieved_ids_to_scores_all_roles_for_q in retrieved_ids_to_scores_all_roles]

    ## for central only
    _, retrieved_ids_to_scores_central_only = retrieve.retrieve(training_df=training_df, testing_df=testing_df,
                                                                no_similar_hypotheses=no_similar_hypotheses,
                                                                no_retrieved_facts=no_retrieved_facts,
                                                                bm25_model=bm25_model, fit_on=fit_on, only_central=True,
                                                                ret_ids_to_scores=True, based_on_ep=True,
                                                                retrieve_training=False)
    retrieved_central_only = [[fact_id for fact_id in retrieved_ids_to_scores_central_only_for_q] for
                              retrieved_ids_to_scores_central_only_for_q in retrieved_ids_to_scores_central_only]

    ## for grounding only
    _, retrieved_ids_to_scores_grounding_only = retrieve.retrieve(training_df=training_df, testing_df=testing_df,
                                                                  no_similar_hypotheses=no_similar_hypotheses,
                                                                  no_retrieved_facts=no_retrieved_facts,
                                                                  bm25_model=bm25_model, fit_on=fit_on,
                                                                  only_grounding=True,
                                                                  ret_ids_to_scores=True, based_on_ep=True,
                                                                  retrieve_training=False)
    retrieved_grounding_only = [[fact_id for fact_id in retrieved_ids_to_scores_grounding_only_for_q] for
                                retrieved_ids_to_scores_grounding_only_for_q in retrieved_ids_to_scores_grounding_only]

    ## for lexglue only
    _, retrieved_ids_to_scores_lexglue_only = retrieve.retrieve(training_df=training_df, testing_df=testing_df,
                                                                no_similar_hypotheses=no_similar_hypotheses,
                                                                no_retrieved_facts=no_retrieved_facts,
                                                                bm25_model=bm25_model, fit_on=fit_on, only_lexglue=True,
                                                                ret_ids_to_scores=True, based_on_ep=True,
                                                                retrieve_training=False)
    retrieved_lexglue_only = [[fact_id for fact_id in retrieved_ids_to_scores_lexglue_only_for_q] for
                              retrieved_ids_to_scores_lexglue_only_for_q in retrieved_ids_to_scores_lexglue_only]

    return retrieved_all_roles, retrieved_central_only, retrieved_grounding_only, retrieved_lexglue_only


def get_relevant_facts(testing_df):
    explanations_corpus = get_dev_explanations_corpus()

    relevant_all_roles = []
    relevant_central_only = []
    relevant_grounding_only = []
    relevant_lexglue_only = []

    question_ids = testing_df["question_id"]
    for id in question_ids:
        relevant_all_for_this_question = []
        relevant_central_only_for_this_question = []
        relevant_grounding_only_for_this_question = []
        relevant_lexglue_only_for_this_question = []

        question_explanations = explanations_corpus[id]["explanation"]
        for fact_id, fact_role in question_explanations.items():
            relevant_all_for_this_question.append(fact_id)
            if get_training_exp_role_from_wtv2_exp_role(fact_role) == GROUNDING:
                relevant_grounding_only_for_this_question.append(fact_id)
            elif fact_role == CENTRAL:
                relevant_central_only_for_this_question.append(fact_id)
            elif fact_role == LEXGLUE:
                relevant_lexglue_only_for_this_question.append(fact_id)

        relevant_all_roles.append(relevant_all_for_this_question)
        relevant_central_only.append(relevant_central_only_for_this_question)
        relevant_grounding_only.append(relevant_grounding_only_for_this_question)
        relevant_lexglue_only.append(relevant_lexglue_only_for_this_question)

    return relevant_all_roles, relevant_central_only, relevant_grounding_only, relevant_lexglue_only


# dev has 296 entries, remember you leave last 200 for testing, don't include them here
def get_dev_explanations_corpus():
    with open(DEV_TATA_JSON_PATH) as json_file:
        explanations_corpus = json.load(json_file)

    dev_df = pd.read_csv(DEV_CSV_DATA_PATH, sep="\t")
    dev_question_ids = set(dev_df["question_id"].tolist())

    dev_reduced_explanations_corpus = {}
    for q_id, q_dict in explanations_corpus.items():
        if q_id in dev_question_ids:
            dev_reduced_explanations_corpus[q_id] = q_dict

    return dev_reduced_explanations_corpus


def average_precision_at_k(retrieved_list, relevant_list):
    assert len(retrieved_list) == len(relevant_list)

    precisions_at_k_list = []
    for i in range(len(retrieved_list)):
        precisions_at_k_list.append(precision_at_k(retrieved_list[i], relevant_list[i]))

    precisions_at_k_matrix = np.array(precisions_at_k_list)
    # average over columns
    return precisions_at_k_matrix.mean(axis=0)


def precision_at_k(retrieved, relevant):
    precisions_at_k = []
    relevant_set = set(relevant)
    for k in range(1, 7):
        retrieved_set_at_k = set(list(retrieved[:k]))
        no_relevant_and_retrieved = len(retrieved_set_at_k.intersection(relevant_set))
        precision_at_k = no_relevant_and_retrieved / k
        precisions_at_k.append(precision_at_k)
    return precisions_at_k


def average_no_facts_copied_from_retrieved_by_bart_chain_retrieve():
    # for each quesiton
    # get no facts copied by bart-chain-retrieve
    # get no facts copied by bart-chain
    # subtract
    # average for all questions

    # get actuals for each question

    actuals_central = []
    actuals_grounding = []
    actuals_lexglue = []

    actuals_for_each_exp_role = pd.read_csv(DEV_CSV_DATA_PATH, sep="\t")["explanation"]
    for i in range(0, len(actuals_for_each_exp_role), 3):
        central_actual = actuals_for_each_exp_role[i]
        actuals_central.append(central_actual)
        grounding_actual = actuals_for_each_exp_role[i + 1]
        actuals_grounding.append(grounding_actual)
        lexglue_actual = actuals_for_each_exp_role[i + 2]
        actuals_lexglue.append(lexglue_actual)

    ##############

    generated_bart_chain_retrieve_central = []
    generated_bart_chain_retrieve_grounding = []
    generated_bart_chain_retrieve_lexglue = []

    df_bart_chain_retrieve = pd.read_csv("./evaluation/BART-chain-retrieve/validation_predictions_vs_actuals.csv")

    generated_all = df_bart_chain_retrieve["Generated"]

    for i in range(len(generated_all)):
        generated_all_arr = generated_all[i].split(EXPLANATORY_ROLES_FACTS_SEP)
        generated_bart_chain_retrieve_central.append(generated_all_arr[0])
        generated_bart_chain_retrieve_grounding.append(generated_all_arr[1])
        generated_bart_chain_retrieve_lexglue.append(generated_all_arr[2])

    ############

    retrieved_central = df_bart_chain_retrieve[CENTRAL_RETRIEVED].tolist()
    retrieved_grounding = df_bart_chain_retrieve[GROUNDING_RETRIEVED].tolist()
    retrieved_lexglue = df_bart_chain_retrieve[LEXGLUE_RETRIEVED].tolist()

    ###########

    generated_bart_chain_central = []
    generated_bart_chain_grounding = []
    generated_bart_chain_lexglue = []

    df_bart_chain = pd.read_csv("./evaluation/BART-chains/validation_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")
    generated_all = df_bart_chain["Generated"]

    for i in range(len(generated_all)):
        generated_all_arr = generated_all[i].split(EXPLANATORY_ROLES_FACTS_SEP)
        generated_bart_chain_central.append(generated_all_arr[0])
        generated_bart_chain_grounding.append(generated_all_arr[1])
        generated_bart_chain_lexglue.append(generated_all_arr[2])


    bart_chain_central_copied = get_no_retrieved_samples_copied_to_output(generated_with_seperater=generated_bart_chain_central,
                                                                          retrieved_with_separator=retrieved_central,
                                                                          ret_facts_sep=CENTRAL_FACTS_SEP, gen_facts_sep=CENTRAL_FACTS_SEP)
    bart_chain_retrieved_central_copied = get_no_retrieved_samples_copied_to_output(generated_with_seperater=generated_bart_chain_retrieve_central,
                                                                                    retrieved_with_separator=retrieved_central,
                                                                                    ret_facts_sep=CENTRAL_FACTS_SEP, gen_facts_sep=CENTRAL_FACTS_SEP)

    bart_chain_grounding_copied = get_no_retrieved_samples_copied_to_output(generated_with_seperater=generated_bart_chain_grounding,
                                                                          retrieved_with_separator=retrieved_grounding,
                                                                          ret_facts_sep=GROUNDING_FACTS_SEP, gen_facts_sep=GROUNDING_FACTS_SEP)
    bart_chain_retrieved_grounding_copied = get_no_retrieved_samples_copied_to_output(generated_with_seperater=generated_bart_chain_retrieve_grounding,
                                                                                    retrieved_with_separator=retrieved_grounding,
                                                                                    ret_facts_sep=GROUNDING_FACTS_SEP, gen_facts_sep=GROUNDING_FACTS_SEP)

    bart_chain_lexglue_copied = get_no_retrieved_samples_copied_to_output(generated_with_seperater=generated_bart_chain_lexglue,
                                                                          retrieved_with_separator=retrieved_lexglue,
                                                                          ret_facts_sep=LEXGLUE_FACTS_SEP, gen_facts_sep=LEXGLUE_FACTS_SEP)
    bart_chain_retrieved_lexglue_copied = get_no_retrieved_samples_copied_to_output(generated_with_seperater=generated_bart_chain_retrieve_lexglue,
                                                                                    retrieved_with_separator=retrieved_lexglue,
                                                                                    ret_facts_sep=LEXGLUE_FACTS_SEP, gen_facts_sep=LEXGLUE_FACTS_SEP)













if __name__ == "__main__":
    fit_on = "hypothesis"
    max_no_retrieved_facts = 7

    train_df = pd.read_csv(TRAINING_CSV_DATA_PATH, sep="\t")
    test_df = pd.read_csv(DEV_CSV_DATA_PATH, sep="\t")

    bm25_model = retrieve.fit_bm25_on_wtv2(training_questions=train_df[fit_on],
                                           training_questions_ids=train_df["question_id"])

    (retrieved_all_roles, retrieved_central_only, retrieved_grounding_only,
     retrieved_lexglue_only) = retrieve_facts_based_on_explanatory_power(bm25_model=bm25_model, training_df=train_df,
                                                                         testing_df=test_df,
                                                                         no_similar_hypotheses=20,
                                                                         no_retrieved_facts=max_no_retrieved_facts,
                                                                         fit_on=fit_on)

    relevant_all_roles, relevant_central_only, relevant_grounding_only, relevant_lexglue_only = get_relevant_facts(
        testing_df=test_df)

    # todo change back
    print("for all:")
    precision_at_k_for_all = average_precision_at_k(retrieved_all_roles, relevant_all_roles)
    print(precision_at_k_for_all)

    print("for central")
    precision_at_k_for_central = average_precision_at_k(retrieved_central_only, relevant_central_only)
    print(precision_at_k_for_central)

    print("for grounding")
    precision_at_k_for_grounding = average_precision_at_k(retrieved_grounding_only, relevant_grounding_only)
    print(precision_at_k_for_grounding)

    print("for lexglue")
    precision_at_k_for_lexglue = average_precision_at_k(retrieved_lexglue_only, relevant_lexglue_only)
    print(precision_at_k_for_lexglue)
