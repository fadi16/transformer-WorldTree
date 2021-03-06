import json

import nltk
import pandas as pd

from retrieve_prompt_generate.utils import Utils
from retrieve_prompt_generate.explanatory_power import ExplanatoryPower
from nltk.corpus import stopwords

from retrieve_prompt_generate.bm25 import BM25
from wtv2_constants import *

FACTS_BANK_JSON_PATH = "./data/v2-proper-data/tablestore_shared.json"
TRAINING_DATA_JSON_PATH = "./data/v2-proper-data/train_set_shared.json"



def fit_bm25_on_wtv2(training_questions, training_questions_ids, role=None):
    facts_bank_dict = get_facts_bank_dict(role=role)

    facts_bank = []
    facts_ids = []
    for fact_id, fact_dict in facts_bank_dict.items():
        facts_ids.append(fact_id)
        facts_bank.append(fact_dict["fact"])

    utils = Utils()
    utils.init_explanation_bank_lemmatizer()

    model = BM25()

    # preprocessing facts bank
    lemmatized_facts_bank = []
    for fact in facts_bank:
        lemmatized_fact = utils.preprocess(fact)
        if lemmatized_fact:
            lemmatized_facts_bank.append(lemmatized_fact)

    # preprocessing the q/a - explanations
    lemmatized_questions = []
    for question in training_questions:
        lemmatized_questions.append(utils.preprocess(question))

    # todo: how does this work
    model.fit(lemmatized_facts_bank, lemmatized_questions, facts_ids, training_questions_ids)
    return model


def get_facts_bank_dict(role=None):
    with open(FACTS_BANK_JSON_PATH) as json_file:
        facts_bank_dict = json.load(json_file)

    if role is None:
        return facts_bank_dict

    explanations_corpus = get_explanations_corpus(role=role)

    # get role only facts
    role_only_facts = set()

    for question_id, question_dict in explanations_corpus.items():
        for fact_id, fact_role in question_dict["explanation"].items():
            role_only_facts.add(fact_id)

    # construct new facts_bank_dict with role only facts
    role_only_facts_bank_dict = dict()
    for fact_id, fact_dict in facts_bank_dict.items():
        if fact_id in role_only_facts:
            role_only_facts_bank_dict[fact_id] = fact_dict

    return role_only_facts_bank_dict


def get_explanations_corpus(role=None):
    with open(TRAINING_DATA_JSON_PATH) as json_file:
        explanations_corpus = json.load(json_file)

    if role is None:
        return explanations_corpus

    for question_id, question_dict in explanations_corpus.items():
        question_dict["explanation"] = dict(
            [(fact_id, get_training_exp_role_from_wtv2_exp_role(fact_role)) for fact_id, fact_role in
             question_dict["explanation"].items() if
             get_training_exp_role_from_wtv2_exp_role(fact_role) == role])
    return explanations_corpus


# todo: retrieving based on similarity - not fully done
def retrieve(training_df, testing_df, no_similar_hypotheses, no_retrieved_facts,
             only_central=False, only_grounding=False, only_lexglue=False, retrieved_facts_sep="????",
             bm25_model=None, fit_on="hypothesis", ret_ids_to_scores=False, based_on_ep=True, retrieve_training=True):

    utils = Utils()
    utils.init_explanation_bank_lemmatizer()

    if bm25_model is None:
        bm25_model = fit_bm25_on_wtv2(training_questions=training_df[fit_on],
                                      training_questions_ids=training_df["question_id"])

    if only_central:
        role = CENTRAL
    elif only_grounding:
        role = GROUNDING
    elif only_lexglue:
        role = LEXGLUE
    else:
        role = None


    facts_bank_dict = get_facts_bank_dict(role=role)
    facts_bank_ids = set(facts_bank_dict.keys())

    explanations_corpus = get_explanations_corpus(role=role)

    EP = ExplanatoryPower(ranker=bm25_model, explanations_corpus=explanations_corpus)

    def get_retrieved_facts(df):
        questions = df[fit_on]
        questions_ids = df["question_id"]
        retrieved_facts = []
        retrieved_facts_ids_to_scores = []
        for question, question_id in zip(questions, questions_ids):
            lemmatized_question = utils.preprocess(question)
            if based_on_ep:
                res = EP.compute(
                    q_id=question_id,
                    query=lemmatized_question,
                    sim_questions_limit=no_similar_hypotheses,
                    facts_limit=no_retrieved_facts
                )
            else:
                res = bm25_model.query(query=[lemmatized_question],
                                       top_k=len(bm25_model.joined_corpus))
                # only keep facts of the required role in results
                new_res = {}
                for fact_id in res:
                    if fact_id in facts_bank_ids:
                        new_res[fact_id] = res[fact_id]
                res = new_res

            retrieved_facts_ids_to_scores.append(res)
            retrieved_facts_for_question = []
            for high_importance_fact_id in res:
                high_importance_fact = facts_bank_dict[high_importance_fact_id]["fact"]
                retrieved_facts_for_question.append(high_importance_fact)
            retrieved_facts.append(" {0} ".format(retrieved_facts_sep).join(retrieved_facts_for_question))

        if ret_ids_to_scores:
            return retrieved_facts_ids_to_scores
        else:
            return retrieved_facts

    # load test data and retrieve facts for each question
    testing_retrieved_facts = get_retrieved_facts(testing_df)
    if retrieve_training:
        training_retrieved_facts = get_retrieved_facts(training_df)
    else:
        training_retrieved_facts = None
    return training_retrieved_facts, testing_retrieved_facts


# todo double check
def sort_facts_based_on_similarity_to_question(bm25_model, fact_ids, lemmatized_question):
    ret_facts_ids_to_scores = bm25_model.query(query=[lemmatized_question],
                             top_k=len(bm25_model.joined_corpus))
    question_sorted_facts_ids = []
    for ret_fact_id in ret_facts_ids_to_scores:
        if ret_fact_id in fact_ids:
            question_sorted_facts_ids.append(ret_fact_id)
            if len(question_sorted_facts_ids) == len(fact_ids):
                break

    return question_sorted_facts_ids


# testing sorting how similar facts are to question / hypothesis
if __name__ == "__main__":

    training_df = pd.read_csv("../data/v2-proper-data/train_data_wed.csv", sep="\t")
    bm25_model = fit_bm25_on_wtv2(training_df)

    with open("../data/v2-proper-data/tablestore_shared.json") as json_file:
        facts_bank_dict = json.load(json_file)

    utils = Utils()
    utils.init_explanation_bank_lemmatizer()

    question = "earth rotates on its axis  best explains why the sun appears to move across the sky every day"
    temp = []
    for word in nltk.word_tokenize(question):
        if not word.lower() in stopwords.words("english"):
            temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)

    facts = bm25_model.query(query=[lemmatized_question], top_k=10)
    for f in facts:
        print(facts_bank_dict[f["id"]]["fact"], " === ", f["score"])
