import json
import random

import pandas as pd

pd.set_option('display.max_columns', 16)

from rich.table import Column, Table
from rich import box
import rich
import pickle

facts_dict = {}

explanatory_role_to_sep = {
    "CENTRAL": " && ",
    "GROUNDING": " $$ ",
    "BACKGROUND": " $$ ",
    "LEXGLUE": " %% "
}

CENTRAL = "CENTRAL"
GROUNDING = "GROUNDING"
BACKGROUND = "BACKGROUND"
LEXGLUE = "LEXGLUE"


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


def construct_data_table_with_explanatory_role_chains(data_json, hypotheses_json):
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

        question_and_answer_for_central = question_and_answer + explanatory_role_to_sep[CENTRAL]
        hypothesis_for_central = hypothesis + explanatory_role_to_sep[CENTRAL]
        central_explanations = [explanations_list[i] for i in range(len(explanations_list)) if
                                fact_explanatory_roles[i] == CENTRAL]
        central_explanations_str = explanatory_role_to_sep[CENTRAL].join(central_explanations)
        central_explanation_types = [explanation_types[i] for i in range(len(explanation_types)) if
                                     fact_explanatory_roles[i] == CENTRAL]
        central_explanation_types_str = explanatory_role_to_sep[CENTRAL].join(central_explanation_types)
        data_table.append(
            [
                question_id,
                question_and_answer_for_central,
                hypothesis_for_central,
                central_explanations_str,
                central_explanation_types_str,
                question_topics,
                major_question_topic
            ]
        )

        question_and_answer_for_grounding = question_and_answer_for_central + central_explanations_str + \
                                            explanatory_role_to_sep[GROUNDING]
        hypothesis_for_grounding = hypothesis_for_central + central_explanations_str + explanatory_role_to_sep[
            GROUNDING]
        grounding_explanations = [explanations_list[i] for i in range(len(explanations_list)) if
                                  fact_explanatory_roles[i] == GROUNDING or fact_explanatory_roles[i] == BACKGROUND]
        grounding_explanations_str = explanatory_role_to_sep[GROUNDING].join(grounding_explanations)
        grounding_explanation_types = [explanation_types[i] for i in range(len(explanation_types)) if
                                       fact_explanatory_roles[i] == GROUNDING or fact_explanatory_roles[
                                           i] == BACKGROUND]
        grounding_explanation_types_str = explanatory_role_to_sep[GROUNDING].join(grounding_explanation_types)
        data_table.append(
            [
                question_id,
                question_and_answer_for_grounding,
                hypothesis_for_grounding,
                grounding_explanations_str,
                grounding_explanation_types_str,
                question_topics,
                major_question_topic
            ]
        )

        question_and_answer_for_lexglue = question_and_answer_for_grounding + grounding_explanations_str + \
                                          explanatory_role_to_sep[LEXGLUE]
        hypothesis_for_lexglue = hypothesis_for_grounding + grounding_explanations_str + explanatory_role_to_sep[
            LEXGLUE]
        lexglue_explanations = [explanations_list[i] for i in range(len(explanations_list)) if
                                fact_explanatory_roles[i] == LEXGLUE]
        lexglue_explanations_str = explanatory_role_to_sep[LEXGLUE].join(lexglue_explanations)
        lexglue_explanation_types = [explanation_types[i] for i in range(len(explanation_types)) if
                                     fact_explanatory_roles[i] == LEXGLUE]
        lexglue_explanation_types_str = explanatory_role_to_sep[LEXGLUE].join(lexglue_explanation_types)
        data_table.append(
            [
                question_id,
                question_and_answer_for_lexglue,
                hypothesis_for_lexglue,
                lexglue_explanations_str,
                lexglue_explanation_types_str,
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

    # dev data
    dev_table, questions_misssing_hypo = construct_data_table("./data/v2-proper-data/dev_set_shared.json",
                                                              "./data/v2-proper-data/hypothesis_dev_v2.json")
    dev_table_chains, _ = construct_data_table_with_explanatory_role_chains("./data/v2-proper-data/dev_set_shared.json",
                                                                            "./data/v2-proper-data/hypothesis_dev_v2.json")

    reduced_dev_table = dev_table[:-200]
    reduced_dev_table_chains = dev_table_chains[:-200]

    df = pd.DataFrame(data=reduced_dev_table, columns=columns)
    df.to_csv("./data/v2-proper-data/dev_data_wed.csv", sep="\t")
    df_chains = pd.DataFrame(data=reduced_dev_table_chains, columns=columns)
    df_chains.to_csv("./data/v2-proper-data/dev_data_wed_chains.csv", sep="\t")

    # training data
    train_table, questions_misssing_hypo = construct_data_table("./data/v2-proper-data/train_set_shared.json",
                                                                "./data/v2-proper-data/hypothesis_train_v2.json")
    train_table_chains, _ = construct_data_table_with_explanatory_role_chains(
        "./data/v2-proper-data/train_set_shared.json",
        "./data/v2-proper-data/hypothesis_train_v2.json")
    df = pd.DataFrame(data=train_table, columns=columns)
    df.to_csv("./data/v2-proper-data/train_data_wed.csv", sep="\t")
    df_chains = pd.DataFrame(data=train_table_chains, columns=columns)
    df_chains.to_csv("./data/v2-proper-data/train_data_wed_chains.csv", sep="\t")

    # testing data
    test_table = dev_table[-200:]
    test_table_chains = dev_table_chains[-200:]

    df = pd.DataFrame(data=test_table, columns=columns)
    df.to_csv("./data/v2-proper-data/test_data_wed.csv", sep="\t")
    df_chains = pd.DataFrame(data=test_table_chains, columns=columns)
    df_chains.to_csv("./data/v2-proper-data/test_data_wed_chains.csv", sep="\t")
