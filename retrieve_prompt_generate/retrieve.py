import json

import nltk
import pandas as pd

from retrieve_prompt_generate.utils import Utils
from retrieve_prompt_generate.explanatory_power import ExplanatoryPower
from nltk.corpus import stopwords

from retrieve_prompt_generate.bm25 import BM25

FACTS_BANK_JSON_PATH = "./data/v2-proper-data/tablestore_shared.json"
TRAINING_DATA_JSON_PATH = "./data/v2-proper-data/train_set_shared.json"

CENTRAL = "CENTRAL"
GROUNDING = "GROUNDING"
BACKGROUND = "BACKGROUND"
LEXGLUE = "LEXGLUE"

def fit_bm25_on_wtv2(training_questions, training_questions_ids):

    with open(FACTS_BANK_JSON_PATH) as json_file:
        facts_bank_dict = json.load(json_file)

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

    model.fit(lemmatized_facts_bank, lemmatized_questions, facts_ids, training_questions_ids)
    return model


def filter_out_non_central(explanations_corpus):
    for question_id, question_dict in explanations_corpus.items():
        question_dict["explanation"] = dict(
            [(fact_id, fact_role) for fact_id, fact_role in question_dict["explanation"].items() if
             fact_role == CENTRAL])
    return explanations_corpus


def filter_out_non_grouding(explanations_corpus):
    for question_id, question_dict in explanations_corpus.items():
        question_dict["explanation"] = dict(
            [(fact_id, fact_role) for fact_id, fact_role in question_dict["explanation"].items() if
             fact_role == GROUNDING or fact_role == BACKGROUND])
    return explanations_corpus


def filter_out_non_lexglue(explanations_corpus):
    for question_id, question_dict in explanations_corpus.items():
        question_dict["explanation"] = dict(
            [(fact_id, fact_role) for fact_id, fact_role in question_dict["explanation"].items() if
             fact_role == LEXGLUE])
    return explanations_corpus


def retrieve(training_df, testing_df, no_similar_hypotheses, no_retrieved_facts,
             only_central=False, only_grounding=False, only_lexglue=False, retrieved_facts_sep="££"):

    bm25_model = fit_bm25_on_wtv2(training_questions=training_df["hypothesis"],
                                  training_questions_ids=training_df["question_id"])

    with open(TRAINING_DATA_JSON_PATH) as json_file:
        explanations_corpus = json.load(json_file)

    if only_central:
        explanations_corpus = filter_out_non_central(explanations_corpus)
    elif only_grounding:
        explanations_corpus = filter_out_non_grouding(explanations_corpus)
    elif only_lexglue:
        explanations_corpus = filter_out_non_lexglue(explanations_corpus)

    EP = ExplanatoryPower(ranker=bm25_model, explanations_corpus=explanations_corpus)

    def get_retrieved_facts(df):
        questions = df["hypothesis"]
        questions_ids = df["question_id"]
        retrieved_facts = []
        for question, question_id in zip(questions, questions_ids):
            lemmatized_question = utils.preprocess(question)
            explanatory_power = EP.compute(
                q_id=question_id,
                query=lemmatized_question,
                sim_questions_limit=no_similar_hypotheses,
                facts_limit=no_retrieved_facts
            )
            retrieved_facts_for_question = []
            for high_exp_power_fact_id in explanatory_power:
                high_exp_power_fact = facts_bank_dict[high_exp_power_fact_id]["fact"]
                retrieved_facts_for_question.append(high_exp_power_fact)
            retrieved_facts.append(" {0} ".format(retrieved_facts_sep).join(retrieved_facts_for_question))
        return retrieved_facts

    # load test data and retrieve facts for each question
    testing_retrieved_facts = get_retrieved_facts(testing_df)
    training_retrieved_facts = get_retrieved_facts(training_df)

    return training_retrieved_facts, testing_retrieved_facts


def sort_facts_based_on_similarity_to_question(bm25_model, fact_ids, lemmatized_question):
    facts = bm25_model.query(query=[lemmatized_question],
                             top_k=len(bm25_model.joined_corpus))
    question_sorted_facts_ids = []
    for fact in facts:
        if fact["id"] in fact_ids:
            question_sorted_facts_ids.append(fact["id"])
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
