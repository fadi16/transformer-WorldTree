import json
import random
import sys
from wtv2_constants import *
import pandas as pd

from retrieve_prompt_generate.utils import Utils

pd.set_option('display.max_columns', 16)

from rich.table import Column, Table
from rich import box
import rich
import pickle
from retrieve_prompt_generate.retrieve import fit_bm25_on_wtv2, sort_facts_based_on_similarity_to_question

facts_dict = {}

def display_df(df):
    table = Table(
        Column("question_id", justify="center"),
        Column("question_and_answer", justify="center"),
        Column("hypothesis", justify="center"),
        Column("explanation", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )
    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    rich.print(table)


def json_to_dict(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return data


def get_explanations_list(fact_ids_list):
    global facts_dict
    if not facts_dict:
        facts_dict = json_to_dict("./data/v2-proper-data/tablestore_shared.json")
    explanations_list = []
    for fact_id in fact_ids_list:
        explanations_list.append(facts_dict[fact_id]["fact"])
    return explanations_list


def get_explanations_and_explanations_types_list(fact_ids_list):
    global facts_dict
    if not facts_dict:
        facts_dict = json_to_dict("./data/v2-proper-data/tablestore_shared.json")
    explanations_list = []
    explanations_types_list = []
    for fact_id in fact_ids_list:
        explanations_list.append(facts_dict[fact_id]["fact"])
        explanations_types_list.append(facts_dict[fact_id]["table_name"])
    return explanations_list, explanations_types_list


# Num of times each topic occurs:
#
# LIFE 846
# EARTH 949
# MAT 532
# SAF 9
# CEL 358
# OTHER 51
# FOR 86
# SCI 45
# ENG 348

# i.e if specific topic is ENG_BLA_BLA, the major topic is ENG
def get_major_topic_from_specific_topic(topic):
    if "_" in topic:
        return topic.split("_")[0]
    else:
        return topic


# this deals with questions that have one or multiple topics
# for questions that have multiple topics:
# we get the major topics from the specific topics
# then we filter out the major topics we're not interested in (don't occur a lot): [FOR, SCI, SAF, OTHER]
# then we choose the major topic with the highest count within the list, if all counts are same choose at random
def get_major_topic_from_topics(topics):
    MOST_FREQUENT_TOPICS = ["LIFE", "EARTH", "MAT", "CEL", "ENG"]
    if "," in topics:
        topics = topics.split(",")
        major_topics = []
        for topic in topics:
            topic = topic.strip()
            major_topic = get_major_topic_from_specific_topic(topic)
            if major_topic in MOST_FREQUENT_TOPICS:
                major_topics.append(major_topic)
        if major_topics:
            random.shuffle(major_topics)
            return max(major_topics, key=major_topics.count)
        else:
            return "None"
    else:
        major_topic = get_major_topic_from_specific_topic(topics)
        if major_topic in MOST_FREQUENT_TOPICS:
            return major_topic
        else:
            return "None"


# contains: question id, question&answer, explanation, explanation type, hypothesis
def construct_data_table(data_json, hypotheses_json):
    question_id_and_answer_key_to_hypothesis = json_to_dict(hypotheses_json)
    data = json_to_dict(data_json)  # dict from question id to: question, answer, fact ids, answer key etc
    data_table = []  # 4D array, containing question id, question&answer, hypothesis, explanations, topics, major topic
    questions_missing_hypothesis = []
    for question_id in data.keys():
        question_data = data[question_id]
        question = question_data["question"]
        answer = question_data["answer"]
        question_topics = ",".join(topics for topics in question_data["topic"])
        major_question_topic = get_major_topic_from_topics(question_topics)
        question_and_answer = question + " " + answer
        answer_key = question_data["answerKey"]
        try:
            hypothesis = question_id_and_answer_key_to_hypothesis[question_id][answer_key]
        except KeyError:
            hypothesis = question_and_answer
            questions_missing_hypothesis.append(question_id)
        fact_ids_list = list(question_data["explanation"].keys())
        explanations_list, explanations_types_list = get_explanations_and_explanations_types_list(fact_ids_list)
        explanations = " $$ ".join(explanations_list)
        explanations_types = " $$ ".join(explanations_types_list)
        data_table.append(
            [
                question_id,
                question_and_answer,
                hypothesis,
                explanations,
                explanations_types,
                question_topics,
                major_question_topic
            ]
        )

    return data_table, questions_missing_hypothesis


def print_data_table(data_json):
    data = json_to_dict(data_json)
    i = 0
    for question_id in data.keys():
        question_data = data[question_id]
        question = question_data["question"]
        answer = question_data["answer"]
        question_topics = ",".join(topics for topics in question_data["topic"])
        question_and_answer = question + " " + answer

        fact_ids_list = []
        fact_explanatory_roles = []
        for id, role in question_data["explanation"].items():
            fact_ids_list.append(id)
            fact_explanatory_roles.append(role)

        explanations_list, explanation_types = get_explanations_and_explanations_types_list(fact_ids_list)
        print("Question ", i)
        print(question_and_answer)
        for explanatory_role, fact in zip(fact_explanatory_roles, explanations_list):
            print("\t({0}) - {1}".format(explanatory_role, fact))
        i += 1


# todo : see what to do
def contains_wrong_explanatory_role(roles):
    for role in roles:
        if role not in allowed_explanatory_roles:
            return True
    return False


def construct_data_table_with_explanatory_role_chains(data_json, hypotheses_json,
                                                      roles_order=[CENTRAL, GROUNDING, LEXGLUE],
                                                      no_dep_on_before=False):
    """
    :param data_json:
    :param hypotheses_json:
    :param roles_order: dictates which roles with appear first in the data: do e start with central or grounding, etc.
    :param no_dep_on_before: do not include the output of the previous step in the input of the current step
    :return:
    """
    question_id_and_answer_key_to_hypothesis = json_to_dict(hypotheses_json)
    questions_missing_hypothesis = []

    data = json_to_dict(data_json)
    data_table = []  # 4D array, containing question id, question&answer, hypothesis, explanations, topics, major topic

    for question_id in data.keys():
        question_data = data[question_id]
        question = question_data["question"]
        answer = question_data["answer"]
        question_topics = ",".join(topics for topics in question_data["topic"])
        major_question_topic = get_major_topic_from_topics(question_topics)
        question_and_answer = question + " " + answer
        answer_key = question_data["answerKey"]
        try:
            hypothesis = question_id_and_answer_key_to_hypothesis[question_id][answer_key]
        except KeyError:
            hypothesis = question_and_answer
            questions_missing_hypothesis.append(question_id)

        fact_ids_list = []
        fact_explanatory_roles = []
        for id, role in question_data["explanation"].items():
            fact_ids_list.append(id)
            fact_explanatory_roles.append(role)

        explanations_list, explanation_types = get_explanations_and_explanations_types_list(fact_ids_list)

        role_explanations_str = ""
        for role in roles_order:
            question_and_answer += ("" if no_dep_on_before else role_explanations_str) + explanatory_role_to_sep[role]
            hypothesis += ("" if no_dep_on_before else role_explanations_str) + explanatory_role_to_sep[role]
            role_explanations = [explanations_list[i] for i in range(len(explanations_list)) if
                                 get_training_exp_role_from_wtv2_exp_role(fact_explanatory_roles[i]) == role]
            role_explanations_str = explanatory_role_to_sep[role].join(role_explanations)
            if not role_explanations_str:
                role_explanations_str = " "

            role_explanation_types = [explanation_types[i] for i in range(len(explanation_types)) if
                                      get_training_exp_role_from_wtv2_exp_role(fact_explanatory_roles[i]) == role]
            role_explanation_types_str = explanatory_role_to_sep[role].join(role_explanation_types)

            data_table.append(
                [
                    question_id,
                    question_and_answer,
                    hypothesis,
                    role_explanations_str,
                    role_explanation_types_str,
                    question_topics,
                    major_question_topic
                ]
            )

    return data_table, questions_missing_hypothesis


def construct_data_table_with_one_fact_per_hop_chains(data_json, hypotheses_json, no_inference_steps=4):

    utils = Utils()
    utils.init_explanation_bank_lemmatizer()

    question_id_and_answer_key_to_hypothesis = json_to_dict(hypotheses_json)
    questions_missing_hypothesis = []

    data = json_to_dict(data_json)

    # use question id and hypotheses data from previously created csv
    previous_table = pd.read_csv("./data/v2-proper-data/train_data_wed.csv", sep="\t")
    bm25_model = fit_bm25_on_wtv2(training_questions_ids=previous_table["question_id"],
                                  training_questions=previous_table["hypothesis"])


    data_table = []  # 4D array, containing question id, question&answer, hypothesis, explanations, topics, major topic

    for question_id in data.keys():
        question_data = data[question_id]
        question = question_data["question"]
        answer = question_data["answer"]
        question_topics = ",".join(topics for topics in question_data["topic"])
        major_question_topic = get_major_topic_from_topics(question_topics)
        question_and_answer = question + " " + answer
        answer_key = question_data["answerKey"]
        try:
            hypothesis = question_id_and_answer_key_to_hypothesis[question_id][answer_key]
        except KeyError:
            hypothesis = question_and_answer
            questions_missing_hypothesis.append(question_id)

        fact_ids_list = []
        fact_explanatory_roles = []
        for id in question_data["explanation"].keys():
            fact_ids_list.append(id)

        explanations_list, _ = get_explanations_and_explanations_types_list(fact_ids_list)

        sorted_fact_ids = sort_facts_based_on_similarity_to_question(bm25_model, fact_ids_list, lemmatized_question=utils.preprocess(hypothesis))

        for id in sorted_fact_ids:
            fact_explanatory_roles.append(question_data["explanation"][id])

        explanations_list, _ = get_explanations_and_explanations_types_list(sorted_fact_ids)

        explanations_str = ""

        last_fact_index = len(explanations_list) - 1

        for i in range(no_inference_steps):
            if i <= last_fact_index:
                fact = explanations_list[i]
            else:
                fact = " <end> "

            if "<end>  $$" not in question_and_answer:
                question_and_answer += explanations_str + " $$ "
            if "<end>  $$" not in hypothesis:
                hypothesis += explanations_str + " $$ "

            explanations_str = fact

            if i == last_fact_index:
                explanations_str += " <end> "

            data_table.append(
                [
                    question_id,
                    question_and_answer,
                    hypothesis,
                    explanations_str,
                    question_topics,
                    major_question_topic
                ]
            )

        # the rest of the facts
        if "<end>  $$" not in question_and_answer:
            question_and_answer += explanations_str + " $$ "
        if "<end>  $$" not in hypothesis:
            hypothesis += explanations_str + " $$ "
        if last_fact_index >= no_inference_steps:
            remaining_facts = explanations_list[no_inference_steps:]
            if len(remaining_facts) > 1:
                explanations_str = " $$ ".join(explanations_list[no_inference_steps:]) + " <end> "
            else:
                explanations_str = remaining_facts[0] + " <end> "
        else:
            explanations_str = " <end> "

        explanations_types_str = " "
        data_table.append(
            [
                question_id,
                question_and_answer,
                hypothesis,
                explanations_str,
                explanations_types_str,
                question_topics,
                major_question_topic
            ]
        )
    return data_table, questions_missing_hypothesis


if __name__ == "__main__":
    columns = ["question_id", "question_and_answer", "hypothesis", "explanation", "explanation_type", "question_topics",
               "major_question_topic"]
    ####################################################################
    # use last 200 samples from dev for test
    ####################################################################
    order = [GROUNDING, CENTRAL, LEXGLUE]
    order_str = "_".join(order)

    # dev data
    dev_table, questions_misssing_hypo = construct_data_table("./data/v2-proper-data/dev_set_shared.json",
                                                              "./data/v2-proper-data/hypothesis_dev_v2.json")
    dev_table_chains, _ = construct_data_table_with_explanatory_role_chains("./data/v2-proper-data/dev_set_shared.json",
                                                                            "./data/v2-proper-data/hypothesis_dev_v2.json",
                                                                            roles_order=order
                                                                            )
    no_inference_steps = 4
    dev_table_inference_chains, _ = construct_data_table_with_one_fact_per_hop_chains("./data/v2-proper-data/dev_set_shared.json",
                                                                                      "./data/v2-proper-data/hypothesis_dev_v2.json",
                                                                                      no_inference_steps=no_inference_steps)

    reduced_dev_table = dev_table[:-200]
    reduced_dev_table_chains = dev_table_chains[:-600]
    reduced_dev_table_inference_chains = dev_table_inference_chains[:(-200 * (no_inference_steps + 1))]

    df = pd.DataFrame(data=reduced_dev_table, columns=columns)
    df.to_csv("./data/v2-proper-data/dev_data_wed.csv", sep="\t")
    df_chains = pd.DataFrame(data=reduced_dev_table_chains, columns=columns)
    df_chains.to_csv("./data/v2-proper-data/dev_data_wed_chains_grounding_first.csv", sep="\t")
    df_inference_chains = pd.DataFrame(data=reduced_dev_table_inference_chains, columns=columns)
    df_inference_chains.to_csv("./data/v2-proper-data/dev_data_wed_inference_chains_{0}.csv".format(no_inference_steps), sep="\t")

    # training data
    train_table, questions_misssing_hypo = construct_data_table("./data/v2-proper-data/train_set_shared.json",
                                                                "./data/v2-proper-data/hypothesis_train_v2.json")
    train_table_chains, _ = construct_data_table_with_explanatory_role_chains(
        "./data/v2-proper-data/train_set_shared.json",
        "./data/v2-proper-data/hypothesis_train_v2.json",
        roles_order=order
    )
    train_table_inference_chains, _ = construct_data_table_with_one_fact_per_hop_chains(
        "./data/v2-proper-data/train_set_shared.json",
        "./data/v2-proper-data/hypothesis_train_v2.json",
        no_inference_steps=no_inference_steps)

    df = pd.DataFrame(data=train_table, columns=columns)
    df.to_csv("./data/v2-proper-data/train_data_wed.csv", sep="\t")
    df_chains = pd.DataFrame(data=train_table_chains, columns=columns)
    df_chains.to_csv("./data/v2-proper-data/train_data_wed_chains_grounding_first.csv", sep="\t")
    df_inference_chains = pd.DataFrame(data=train_table_inference_chains, columns=columns)
    df_inference_chains.to_csv("./data/v2-proper-data/train_data_wed_inference_chains_{0}.csv".format(no_inference_steps), sep="\t")

    # testing data
    test_table = dev_table[-200:]
    test_table_chains = dev_table_chains[-600:]
    test_table_inference_chains = dev_table_inference_chains[-200 * (no_inference_steps + 1):]

    df = pd.DataFrame(data=test_table, columns=columns)
    df.to_csv("./data/v2-proper-data/test_data_wed.csv", sep="\t")
    df_chains = pd.DataFrame(data=test_table_chains, columns=columns)
    df_chains.to_csv("./data/v2-proper-data/test_data_wed_chains_grounding_first.csv", sep="\t")
    df_inference_chains = pd.DataFrame(data=test_table_inference_chains, columns=columns)
    df_inference_chains.to_csv("./data/v2-proper-data/test_data_wed_inference_chains_{0}.csv".format(no_inference_steps), sep="\t")