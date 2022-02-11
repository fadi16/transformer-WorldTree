import json

from utils import Utils
from explanatory_power import ExplanatoryPower
import pandas as pd

from bm25 import BM25

TRAINING_DATA_CSV_PATH = "../data/v2-proper-data/train_data_wed.csv"
FACTS_BANK_JSON_PATH = "../data/v2-proper-data/tablestore_shared.json"
TRAINING_DATA_JSON_PATH = "../data/v2-proper-data/train_set_shared.json"
TESTING_DATA_CSV_PATH = "../data/v2-proper-data/test_data_wed.csv"
DEV_DATA_CSV_PATH = "../data/v2-proper-data/dev_data_wed.csv"


def fit_bm25_on_wtv2(utils, facts_bank, training_questions, facts_ids, training_questions_ids):
    model = BM25()

    # preprocessing facts bank
    lemmatized_facts_bank = []
    for fact in facts_bank:
        lemmatized_fact = utils.preprocess_fact(fact)
        if lemmatized_fact:
            lemmatized_facts_bank.append(lemmatized_fact)

    # preprocessing the q/a - explanations
    lemmatized_questions = []
    for question in training_questions:
        lemmatized_questions.append(utils.preprocess_question(question))

    model.fit(lemmatized_facts_bank, lemmatized_questions, facts_ids, training_questions_ids)
    return model


if __name__ == "__main__":
    training_df = pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")
    training_questions = training_df["hypothesis"]
    training_questions_ids = training_df["question_id"]

    with open(TRAINING_DATA_JSON_PATH) as json_file:
        explanations_corpus = json.load(json_file)

    with open(FACTS_BANK_JSON_PATH) as json_file:
        facts_bank_dict = json.load(json_file)

    facts_bank = []
    facts_ids = []
    for fact_id, fact_dict in facts_bank_dict.items():
        facts_ids.append(fact_id)
        facts_bank.append(fact_dict["fact"])

    utils = Utils()
    utils.init_explanation_bank_lemmatizer()

    bm25_model = fit_bm25_on_wtv2(utils=utils,
                                  facts_bank=facts_bank,
                                  facts_ids=facts_ids,
                                  training_questions=training_questions,
                                  training_questions_ids=training_questions_ids)

    EP = ExplanatoryPower(ranker=bm25_model, explanations_corpus=explanations_corpus)

    Q = 3  # limit for number of similar hypothesis
    QK = 5  # limit for retrieved facts


    def retrieve(df):
        questions = df["hypothesis"]
        questions_ids = df["question_id"]
        retrieved_facts = []
        for question, question_id in zip(questions, questions_ids):
            lemmatized_question = utils.preprocess_question(question, remove_stopwords=True)
            explanatory_power = EP.compute(
                q_id=question_id,
                query=lemmatized_question,
                sim_questions_limit=Q,
                facts_limit=QK
            )
            retrieved_facts_for_question = []
            for high_exp_power_fact_id in explanatory_power:
                high_exp_power_fact = facts_bank_dict[high_exp_power_fact_id]["fact"]
                retrieved_facts_for_question.append(high_exp_power_fact)
            retrieved_facts.append(" ££ ".join(retrieved_facts_for_question))
        df["retrieved_facts"] = retrieved_facts
        return df


    # load test data and retrieve facts for each question
    testing_df = pd.read_csv(TESTING_DATA_CSV_PATH, "\t")
    testing_df = retrieve(testing_df)
    testing_df.to_csv(".".join(TESTING_DATA_CSV_PATH.split(".")[:-1]) + "_with_retrieved_exps.csv", sep="\t")

    val_df = pd.read_csv(DEV_DATA_CSV_PATH, "\t")
    val_df = retrieve(val_df)
    val_df.to_csv(".".join(DEV_DATA_CSV_PATH.split(".")[:-1]) + "_with_retrieved_exps.csv", sep="\t")

    training_df = pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")
    training_df = retrieve(training_df)
    training_df.to_csv(".".join(TRAINING_DATA_CSV_PATH.split(".")[:-1]) + "_with_retrieved_exps.csv", sep="\t")
