import json
import pandas as pd

pd.set_option('display.max_columns', 16)

from rich.table import Column, Table
from rich import box
import rich

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

# contains: question id, question&answer, explanation, hypothesis
def construct_data_table(data_json, hypotheses_json):
    question_id_and_answer_key_to_hypothesis = json_to_dict(hypotheses_json)
    data = json_to_dict(data_json) # dict from question id to: question, answer, fact ids, answer key etc
    data_table = [] # 4D array, containing question id, question&answer, hypothesis, explanations
    questions_missing_hypothesis = []
    for question_id in data.keys():
        question_data = data[question_id]
        question = question_data["question"]
        answer = question_data["answer"]
        question_and_answer = question + " " + answer
        answer_key = question_data["answerKey"]
        try:
            hypothesis = question_id_and_answer_key_to_hypothesis[question_id][answer_key]
        except KeyError:
            hypothesis = question_and_answer
            questions_missing_hypothesis.append(question_id)
        fact_ids_list = list(question_data["explanation"].keys())
        explanations_list = get_explanations_list(fact_ids_list)
        explanations = "$$ ".join(explanations_list)
        data_table.append(
            [
                question_id,
                question_and_answer,
                hypothesis,
                explanations
            ]
        )

    return data_table, questions_missing_hypothesis

if __name__ == "__main__":
    columns = ["question_id", "question_and_answer", "hypothesis", "explanation"]
    ####################################################################
    # use last 200 samples from dev for test
    ####################################################################

    # dev data
    dev_table, questions_misssing_hypo = construct_data_table("./data/v2-proper-data/dev_set_shared.json", "./data/v2-proper-data/hypothesis_dev_v2.json")
    reduced_dev_table = dev_table[:-200]
    df = pd.DataFrame(data=reduced_dev_table, columns=columns)
    df.to_csv("./data/v2-proper-data/dev_data.csv", sep="\t")

    # training data
    train_table, questions_misssing_hypo = construct_data_table("./data/v2-proper-data/train_set_shared.json", "./data/v2-proper-data/hypothesis_train_v2.json")
    df = pd.DataFrame(data=train_table, columns=columns)
    df.to_csv("./data/v2-proper-data/train_data.csv", sep="\t")

    # testing data
    test_table = dev_table[:200]
    df = pd.DataFrame(data=test_table, columns=columns)
    df.to_csv("./data/v2-proper-data/test_data.csv", sep="\t")