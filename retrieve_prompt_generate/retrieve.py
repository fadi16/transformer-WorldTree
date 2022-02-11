import json

from retrieve_prompt_generate.utils import Utils
from retrieve_prompt_generate.explanatory_power import ExplanatoryPower

from retrieve_prompt_generate.bm25 import BM25

FACTS_BANK_JSON_PATH = "./data/v2-proper-data/tablestore_shared.json"
TRAINING_DATA_JSON_PATH = "./data/v2-proper-data/train_set_shared.json"


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


def retrieve(training_df, testing_df, no_similar_hypotheses, no_retrieved_facts):
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

    def get_retrieved_facts(df):
        questions = df["hypothesis"]
        questions_ids = df["question_id"]
        retrieved_facts = []
        for question, question_id in zip(questions, questions_ids):
            lemmatized_question = utils.preprocess_question(question, remove_stopwords=True)
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
            retrieved_facts.append(" ££ ".join(retrieved_facts_for_question))
        return retrieved_facts


    # load test data and retrieve facts for each question
    testing_retrieved_facts = get_retrieved_facts(testing_df)
    training_retrieved_facts = get_retrieved_facts(training_df)

    return training_retrieved_facts, testing_retrieved_facts
