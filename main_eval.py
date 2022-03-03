import sys
import heapq
from typing import List

import nltk

nltk.download('stopwords')
# nltk.download("punc")
import pandas as pd
import numpy as np
from datasets import load_metric
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

###############################################
## todo: change file path
###############################################
DEV_PREDICTIONS_CSV_PATH = "evaluation/BART-retrieve-prompt/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv"  # "evaluation/t5-plain/validation_predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting.csv"  #"evaluation/BART-lr-3e-5/test_predictions_vs_actuals_with_BLEURT_scores.csv"  # "outputs/dummy_predicions_with_BLEURT_scores.csv"  # "./evaluation/predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting.csv" #"./evaluation/predictions_vs_actuals-t5-plain-from-hypothesis-with-data-splitting.csv"
# DEV_PREDICTIONS_CSV_PATH = "evaluation/t5-plain/validation_predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting_with_BLEURT_scores.csv"  # "evaluation/t5-plain/validation_predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting.csv"  #"evaluation/BART-lr-3e-5/test_predictions_vs_actuals_with_BLEURT_scores.csv"  # "outputs/dummy_predicions_with_BLEURT_scores.csv"  # "./evaluation/predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting.csv" #"./evaluation/predictions_vs_actuals-t5-plain-from-hypothesis-with-data-splitting.csv"
TRAINING_DATA_CSV_PATH = "data/v2-proper-data/train_data_wed.csv"

num_of_best_worst_explanations = 15
STOP_WORDS = stopwords.words("english")
###############################################
BLEURT_SCORES = "bleurt_scores"
FIGURE_COUNTER = 0

stemmer = PorterStemmer()
# tokenizer to only allow alpha-neumeric characters
regexp_tokenizer = RegexpTokenizer(r'\w+')
bleurt_metric = None

MAIN_FACTS_SEP = "$$"
EXPLANATORY_ROLES_FACTS_SEP = "||"
CENTRAL_FACTS_SEP = "&&"
GROUNDING_FACTS_SEP = "$$"
BACKGROUND_FACTS_SEP = "$$"
LEXGLUE_FACTS_SEP = "%%"
QUESTION_AND_RETRIEVED_FACTS_SEP = "@@"
RETRIEVED_FACTS_SEP = "££"
START_SEP = "<start>"
END_SEP = "<end>"

ALL_FACTS_SEPARATORS = {
    MAIN_FACTS_SEP, EXPLANATORY_ROLES_FACTS_SEP, CENTRAL_FACTS_SEP,
    GROUNDING_FACTS_SEP, BACKGROUND_FACTS_SEP, LEXGLUE_FACTS_SEP,
    QUESTION_AND_RETRIEVED_FACTS_SEP, RETRIEVED_FACTS_SEP, END_SEP
}


def show_plots():
    plt.show()


def get_figure():
    global FIGURE_COUNTER
    f = plt.figure(FIGURE_COUNTER)
    FIGURE_COUNTER += 1
    return f

def pprint_chain_retrieve_results(generated_csv_path):
    actuals = []

    actuals_for_each_exp_role = pd.read_csv("./data/v2-proper-data/test_data_wed_chains.csv", sep="\t")["explanation"]
    for i in range(0, len(actuals_for_each_exp_role), 3):
        central_actual = actuals_for_each_exp_role[i]
        grounding_actual = actuals_for_each_exp_role[i+1]
        lexglue_actual = actuals_for_each_exp_role[i+2]
        actuals.append("||".join([central_actual, grounding_actual, lexglue_actual]))


    df = pd.read_csv(generated_csv_path)
    questions = df["Questions"]
    #actuals = df["Actual Text"]
    generated = df["Generated Text"]
    central_retrieved = df["CENTRAL_RETRIEVED"]
    grounding_retrieved = df["GROUNDING_RETRIEVED"]
    lexglue_retrieved = df["LEXGLUE_RETRIEVED"]
    scores = df[BLEURT_SCORES]

    for i in range(len(questions)):
        question_generated_list = generated[i].split(EXPLANATORY_ROLES_FACTS_SEP)
        question_actual_list = actuals[i].split(EXPLANATORY_ROLES_FACTS_SEP)

        print("Question - Score ({0})".format(scores[i]))
        print(questions[i])

        print("Actual Centrals:")
        question_actual_centrals = question_actual_list[0].split(CENTRAL_FACTS_SEP)
        for j, exp in enumerate(question_actual_centrals):
            print("\t", j, ".\t", exp)

        print("Retrieved Centrals:")
        question_retrieved_centrals = central_retrieved[i].split(CENTRAL_FACTS_SEP)
        for j, exp in enumerate(question_retrieved_centrals):
            print("\t", j, ".\t", exp)

        print("Generated Centrals:")
        question_generated_centrals = question_generated_list[0].split(CENTRAL_FACTS_SEP)
        for j, exp in enumerate(question_generated_centrals):
            print("\t", j, ".\t", exp)

        print("Actual Groundings:")
        question_actual_grounding = question_actual_list[1].split(GROUNDING_FACTS_SEP)
        for j, exp in enumerate(question_actual_grounding):
            print("\t", j, ".\t", exp)

        print("Retrieved Grounding:")
        question_retrieved_grounding = grounding_retrieved[i].split(GROUNDING_FACTS_SEP)
        for j, exp in enumerate(question_retrieved_grounding):
            print("\t", j, ".\t", exp)

        print("Generated Grounding:")
        question_generated_grounding = question_generated_list[1].split(GROUNDING_FACTS_SEP)
        for j, exp in enumerate(question_generated_grounding):
            print("\t", j, ".\t", exp)

        print("Actual Lexglue:")
        question_actual_lexglue = question_actual_list[2].split(LEXGLUE_FACTS_SEP)
        for j, exp in enumerate(question_actual_lexglue):
            print("\t", j, ".\t", exp)

        print("Retrieved Lexglue:")
        question_retrieved_lexglue = lexglue_retrieved[i].split(LEXGLUE_FACTS_SEP)
        for j, exp in enumerate(question_retrieved_lexglue):
            print("\t", j, ".\t", exp)

        print("Generated Lexglue:")
        question_generated_lexglue = question_generated_list[2].split(LEXGLUE_FACTS_SEP)
        for j, exp in enumerate(question_generated_lexglue):
            print("\t", j, ".\t", exp)

        print("==" * 15)


def pprint_best_worst_results(results):
    for result in results:
        i = 0
        question = result["question"]
        if "@@" in question:
            question_and_retrieved_input = question.split("@@")
            question = question_and_retrieved_input[0]
            retrieved_facts = question_and_retrieved_input[1].split("££")
            print("Question: ", question)
            print("Retrieved facts: ")
            i = 0
            for fact in retrieved_facts:
                print("\t", i, ".\t", fact)
                i += 1
        else:
            print("Question: ", question)

        reference_explanations = result["reference"].split(".")
        print("Reference Explanations:")
        i = 0
        for exp in reference_explanations:
            print("\t", i, ".\t", exp)
            i += 1

        print("Generated Explanations:")
        generated_explanations = result["generated"].split(".")
        i = 0
        for exp in generated_explanations:
            print("\t", i, ".\t", exp)
            i += 1
        print("Score:", result["score"])
        print("=" * 40)


def no_hops_in_reference_vs_score(no_hops_reference, scores):
    mean_score = np.mean(scores)
    no_hops_to_scores = {}
    for i in range(len(no_hops_reference)):
        if no_hops_reference[i] in no_hops_to_scores:
            no_hops_to_scores[no_hops_reference[i]].append(scores[i])
        else:
            no_hops_to_scores[no_hops_reference[i]] = [scores[i]]

    hop_numbers = sorted(list(no_hops_to_scores.keys()))
    average_scores = [np.mean(no_hops_to_scores[hop_no]) for hop_no in hop_numbers]

    figure = get_figure()
    plt.plot(hop_numbers, average_scores, marker="o", linestyle="dashed")
    # add line at mean
    plt.plot(hop_numbers, [mean_score] * len(hop_numbers))
    plt.title("No. Explanations in reference vs Score")
    plt.xlabel("no. explanations in ref")
    plt.ylabel("bleurt scores")
    figure.show()



def evaluate(metric_key: str, questions, references, generated, best_and_worst=True):
    global bleurt_metric
    if metric_key == "bleurt":
        if not bleurt_metric:
            bleurt_metric = load_metric('bleurt', "bleurt-large-512")
    else:
        raise Exception()

    bleurt_metric.add_batch(predictions=generated, references=references)
    c = bleurt_metric.compute()
    scores = np.array(c["scores"])
    scores_mean = np.mean(scores)

    if not best_and_worst:
        return scores, scores_mean, None, None

    questions = np.array(questions)
    references = np.array(references)
    generated = np.array(generated)

    # for best explanations
    indicies_of_best_explanations = np.argpartition(scores, -num_of_best_worst_explanations)[
                                    -num_of_best_worst_explanations:]
    questions_with_best_explanations = questions[indicies_of_best_explanations]
    actual_explanations_for_best_explanations = references[indicies_of_best_explanations]
    best_explanations = generated[indicies_of_best_explanations]
    scores_of_best_explanations = scores[indicies_of_best_explanations]
    best_explanations_df = pd.DataFrame({
        "question": questions_with_best_explanations,
        "reference": actual_explanations_for_best_explanations,
        "generated": best_explanations,
        "score": scores_of_best_explanations
    })

    print("== For {0} ==".format(metric_key))
    print("Mean Score = " + str(scores_mean))
    print("best explanations:")
    best_explanations_dict = best_explanations_df.to_dict("record")
    pprint_best_worst_results(best_explanations_dict)

    # for worst explanations
    indicies_of_worst_explanations = np.argpartition(scores, num_of_best_worst_explanations)[
                                     :num_of_best_worst_explanations]
    questions_with_worst_explanations = questions[indicies_of_worst_explanations]
    actual_explanations_for_worst_explanations = references[indicies_of_worst_explanations]
    worst_explanations = generated[indicies_of_worst_explanations]
    scores_of_worst_explanations = scores[indicies_of_worst_explanations]

    worst_explanations_df = pd.DataFrame({
        "question": questions_with_worst_explanations,
        "reference": actual_explanations_for_worst_explanations,
        "generated": worst_explanations,
        "score": scores_of_worst_explanations
    })

    print("worst explanations:")
    worst_explanations_dict = worst_explanations_df.to_dict("record")
    pprint_best_worst_results(worst_explanations_dict)
    print("==========================================================")

    return scores, scores_mean, best_explanations_df, worst_explanations_df


def get_bow_of_fact(fact):
    tokenized_fact = regexp_tokenizer.tokenize(fact)
    return set(
        stemmer.stem(word.lower().strip()) for word in tokenized_fact if
        word.lower().strip() not in STOP_WORDS and word != "" and not word.isspace()
    )


def no_facts_in_reference_vs_no_facts_in_generated(references_with_separator, generated_with_separator,
                                                   show_total_no_generated_facts=True,
                                                   show_no_generated_facts_in_reference=True,
                                                   show_no_repeatedly_generated_facts=True):
    # repeated facts are only counted once
    NO_FACTS_OCCURRING_IN_REF = "no_facts_occurring_in_reference"
    NO_REPEATED_FACTS = "no_repeated_facts"
    NO_GENERATED_FACTS = "no_generated_facts"
    MEAN_NO_FACTS_OCCURRING_IN_REF = "mean_no_facts_occurring_in_reference"
    MEAN_NO_REPEATED_FACTS = "mean_no_repeated_facts"
    MEAN_NO_GENERATED_FACTS = "mean_no_generated_facts"

    no_reference_facts_to_stats = {}

    for i in range(len(references_with_separator)):
        reference_facts = references_with_separator[i].split(MAIN_FACTS_SEP)
        reference_facts_bows = [get_bow_of_fact(ref_fact) for ref_fact in reference_facts]

        generated_facts = generated_with_separator[i].split(MAIN_FACTS_SEP)
        generated_facts_bows = [get_bow_of_fact(gen_fact) for gen_fact in generated_facts]

        # calculate number of repeated generated facts
        unique_gen_facts_bows = []
        for gen_fact_bow in generated_facts_bows:
            if gen_fact_bow not in unique_gen_facts_bows:
                unique_gen_facts_bows.append(gen_fact_bow)

        no_gen_facts_in_ref = 0
        for gen_fact_bow in unique_gen_facts_bows:
            if gen_fact_bow in reference_facts_bows:
                no_gen_facts_in_ref += 1

        no_repeated_gen_facts = len(generated_facts_bows) - len(unique_gen_facts_bows)
        no_generated_facts = len(generated_facts)

        no_reference_facts = len(reference_facts)

        if no_reference_facts in no_reference_facts_to_stats:
            no_reference_facts_to_stats[no_reference_facts][NO_FACTS_OCCURRING_IN_REF].append(no_gen_facts_in_ref)
            no_reference_facts_to_stats[no_reference_facts][NO_REPEATED_FACTS].append(no_repeated_gen_facts)
            no_reference_facts_to_stats[no_reference_facts][NO_GENERATED_FACTS].append(no_generated_facts)
        else:
            no_reference_facts_to_stats[no_reference_facts] = {
                NO_FACTS_OCCURRING_IN_REF: [no_gen_facts_in_ref],
                NO_REPEATED_FACTS: [no_repeated_gen_facts],
                NO_GENERATED_FACTS: [no_generated_facts]
            }

    for k in no_reference_facts_to_stats.keys():
        no_reference_facts_to_stats[k][MEAN_NO_FACTS_OCCURRING_IN_REF] = np.mean(
            no_reference_facts_to_stats[k][NO_FACTS_OCCURRING_IN_REF])
        no_reference_facts_to_stats[k][MEAN_NO_REPEATED_FACTS] = np.mean(
            no_reference_facts_to_stats[k][NO_REPEATED_FACTS])
        no_reference_facts_to_stats[k][MEAN_NO_GENERATED_FACTS] = np.mean(
            no_reference_facts_to_stats[k][NO_GENERATED_FACTS])

    reference_facts_numbers = sorted(list(no_reference_facts_to_stats.keys()))
    no_gen_facts_occurring_in_ref_mean = [no_reference_facts_to_stats[n][MEAN_NO_FACTS_OCCURRING_IN_REF] for n in
                                          reference_facts_numbers]
    no_repeated_gen_facts = [no_reference_facts_to_stats[n][MEAN_NO_REPEATED_FACTS] for n in reference_facts_numbers]
    no_gen_facts = [no_reference_facts_to_stats[n][MEAN_NO_GENERATED_FACTS] for n in reference_facts_numbers]

    figure = get_figure()

    if show_no_generated_facts_in_reference:
        plt.plot(reference_facts_numbers, no_gen_facts_occurring_in_ref_mean, marker="o", linestyle="dashed",
                 label="No. Generated Facts Occurring in Reference")
    if show_no_repeatedly_generated_facts:
        plt.plot(reference_facts_numbers, no_repeated_gen_facts, marker="o", linestyle="solid",
                 label="No. Repeated Generated Facts")
    if show_total_no_generated_facts:
        plt.plot(reference_facts_numbers, no_gen_facts, marker="o", linestyle="solid",
                 label="Total No. Generated Facts")

    plt.plot(reference_facts_numbers, reference_facts_numbers, label="best case")

    plt.legend(loc="upper left")
    plt.title("No. Reference Facts vs Generated Facts statistics")
    plt.xlabel("no. reference facts")
    plt.ylabel("no. generated facts")
    figure.show()

    return no_reference_facts_to_stats


# 2 facts are the same if their BOWs without stopwords are the same
def no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator, generated_with_separator):
    # repeated facts are only counted once
    NO_FACTS_OCCURRING_IN_REF = "no_facts_occurring_in_reference"
    NO_REPEATED_FACTS = "no_repeated_facts"
    NO_GENERATED_FACTS = "no_generated_facts"
    MEAN_NO_FACTS_OCCURRING_IN_REF = "mean_no_facts_occurring_in_reference"
    MEAN_NO_REPEATED_FACTS = "mean_no_repeated_facts"
    MEAN_NO_GENERATED_FACTS = "mean_no_generated_facts"

    no_gen_facts_to_stats = {}

    for i in range(len(references_with_seperator)):
        reference_facts = references_with_seperator[i].lower().replace(";", " ").split("$$")
        reference_facts[:] = [ref_fact.strip() for ref_fact in reference_facts]
        reference_facts_bows = [get_bow_of_fact(ref_fact) for ref_fact in reference_facts]

        generated_facts = generated_with_separator[i].lower().replace(";", " ").split("$$")
        generated_facts[:] = [gen_fact.strip() for gen_fact in generated_facts]
        generated_facts_bows = [get_bow_of_fact(gen_fact) for gen_fact in generated_facts]

        # calculate number of repeated generated facts
        unique_gen_facts_bows = []
        for gen_fact_bow in generated_facts_bows:
            if gen_fact_bow not in unique_gen_facts_bows:
                unique_gen_facts_bows.append(gen_fact_bow)

        no_gen_facts_in_ref = 0
        for gen_fact_bow in unique_gen_facts_bows:
            if gen_fact_bow in reference_facts_bows:
                no_gen_facts_in_ref += 1

        no_repeated_gen_facts = len(generated_facts_bows) - len(unique_gen_facts_bows)
        no_generated_facts = len(generated_facts)

        if no_generated_facts in no_gen_facts_to_stats:
            no_gen_facts_to_stats[no_generated_facts][NO_FACTS_OCCURRING_IN_REF].append(no_gen_facts_in_ref)
            no_gen_facts_to_stats[no_generated_facts][NO_REPEATED_FACTS].append(no_repeated_gen_facts)
        else:
            no_gen_facts_to_stats[no_generated_facts] = {
                NO_FACTS_OCCURRING_IN_REF: [no_gen_facts_in_ref],
                NO_REPEATED_FACTS: [no_repeated_gen_facts]
            }

    for k in no_gen_facts_to_stats.keys():
        no_gen_facts_to_stats[k][MEAN_NO_FACTS_OCCURRING_IN_REF] = np.mean(
            no_gen_facts_to_stats[k][NO_FACTS_OCCURRING_IN_REF])
        no_gen_facts_to_stats[k][MEAN_NO_REPEATED_FACTS] = np.mean(no_gen_facts_to_stats[k][NO_REPEATED_FACTS])

    generated_facts_numbers = sorted(list(no_gen_facts_to_stats.keys()))
    no_gen_facts_occurring_in_ref_mean = [no_gen_facts_to_stats[n][MEAN_NO_FACTS_OCCURRING_IN_REF] for n in
                                          generated_facts_numbers]
    no_repeated_gen_facts = [no_gen_facts_to_stats[n][MEAN_NO_REPEATED_FACTS] for n in generated_facts_numbers]

    figure = get_figure()

    plt.plot(generated_facts_numbers, no_gen_facts_occurring_in_ref_mean, marker="o", linestyle="dashed",
             label="No. Generated Facts Occurring in Reference")
    plt.plot(generated_facts_numbers, no_repeated_gen_facts, marker="o", linestyle="solid",
             label="No. Repeated Generated Facts")
    plt.legend(loc="upper left")
    plt.title("No. Generated Facts vs No. Generated Facts Occurring in Reference and No. Repeated Generated Facts")
    plt.xlabel("no. generated facts")
    figure.show()

    return no_gen_facts_to_stats


# the higher the score the more similar s1 and s2
def jaccard_similarity(s1, s2):
    """
    J(A, B) = |A intersection B| / (|A| + |B| - |A intersection B|)
    """
    s1_bow = get_bow_of_fact(s1)
    s2_bow = get_bow_of_fact(s2)

    s1_bow_inter_s2_bow = s1_bow.intersection(s2_bow)
    score = len(s1_bow_inter_s2_bow) / (len(s1_bow) + len(s2_bow) - len(s1_bow_inter_s2_bow))
    return score * 100


# TODO: need a new way of passing questions_and_answers with retrieval strategy, pass original df before modificaiton
def similarity_score_for_QnA_and_reference_vs_BLEURT_score_of_generated_explanation(questions_and_answers, references,
                                                                                    scores,
                                                                                    similarity_measure,
                                                                                    similarity_step):
    assert len(questions_and_answers) == len(references) == len(scores)
    mean_score = np.mean(scores)

    similarity_to_score = {}
    similarities_to_show = np.array([i for i in range(0, 100, similarity_step)])
    for similarity_to_show in similarities_to_show:
        similarity_to_score[similarity_to_show] = []

    for i in range(len(questions_and_answers)):
        similarity = similarity_measure(questions_and_answers[i], references[i])

        similarity_diff = abs(similarity - similarities_to_show)
        closest_similarity_index = np.argmin(similarity_diff)
        closest_similarity = similarities_to_show[closest_similarity_index]

        similarity_to_score[closest_similarity].append(scores[i])

    # remove similarities that have no scores associated to them
    similarities = sorted(sim for sim in similarity_to_score.keys() if similarity_to_score[sim] != [])
    scores = [np.mean(similarity_to_score[s]) for s in similarities]
    figure = get_figure()

    plt.plot(similarities, scores, marker="o", linestyle="dashed", label="bluert score of generated senteneces")
    plt.plot(similarities, [mean_score for _ in similarities], label="mean bluert score")
    plt.title("Similarity Score Between Q/A and Golden Reference vs BLEURT Score of Generated Sentence")
    plt.xlabel("similarity between Q/A and golden reference - " + str(
        similarity_measure.__name__) + ", with similarity_step = " + str(similarity_step))
    plt.ylabel("bleurt score of generated sentence")
    plt.legend(loc="upper left")
    figure.show()


# todo: better metric for similarity??
def average_similarity_of_each_test_QnA_to_n_closest_QnAs_in_training_set_vs_no_test_samples_with_this_score(
        qna_testing_set, qna_training_set,
        ns: List[int], similarity_measure):
    # a map between an average similarity between a test QnA and the QnAs from the training set, and the number of those test QnAs with that average similarity
    average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity = {}
    for test_qna in qna_testing_set:
        current_qna_similarity_scores = []
        for train_qna in qna_training_set:
            current_qna_similarity_scores.append(similarity_measure(test_qna, train_qna))

        for n in ns:

            if n not in average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity:
                average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n] = {}

            # get n best scores
            n_most_similar = heapq.nlargest(n, current_qna_similarity_scores)
            average_similarity = int(np.mean(n_most_similar))

            if average_similarity in average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[
                n]:
                average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n][
                    average_similarity] += 1
            else:
                average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n][
                    average_similarity] = 1

    for n in ns:
        figure = get_figure()

        average_similarities = sorted(
            list(average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n].keys()))
        counts = [average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n][avr_sim] for
                  avr_sim in average_similarities]

        plt.plot(average_similarities, counts, marker="o", markersize=5, linestyle="dashed", label="n = {0}".format(n))

        plt.legend(loc="upper left")
        plt.title(
            "Average Similarity Score Between Test QnA And 'n' Closest Training QnAs vs No. of Test QnAs Having That Score".format(
                str(n)))
        plt.xlabel("average similarity between test QnA and 'n' closest training QnAs - " + str(
            similarity_measure.__name__) + " / more means more similar")
        plt.ylabel("number of test QnAs")
        figure.show()


def average_similarity_between_test_and_train_samples_vs_bluert_score(qna_testing_set, qna_training_set, scores, n,
                                                                      similarity_measure, similarity_step):
    mean_score = np.mean(scores)
    average_similarities_to_bleu_scores = {}
    similarities_to_show = np.array([i for i in range(0, 100, similarity_step)])
    for similarity_to_show in similarities_to_show:
        average_similarities_to_bleu_scores[similarity_to_show] = []

    for i in range(len(qna_testing_set)):
        similarities = []
        for training_qna in qna_training_set:
            similarities.append(similarity_measure(qna_testing_set[i], training_qna))
        average_n_most_similar = np.mean(heapq.nlargest(n, similarities))

        similarity_diff = abs(average_n_most_similar - similarities_to_show)
        closest_similarity_index = np.argmin(similarity_diff)
        closest_similarity = similarities_to_show[closest_similarity_index]
        average_similarities_to_bleu_scores[closest_similarity].append(scores[i])

    figure = get_figure()

    similarity_scores = sorted(
        sim for sim in average_similarities_to_bleu_scores.keys() if average_similarities_to_bleu_scores[sim] != [])
    bluert_scores = [np.mean(average_similarities_to_bleu_scores[similarity_score]) for similarity_score in
                     similarity_scores]
    plt.plot(similarity_scores, bluert_scores, marker="o", markersize=5, linestyle="solid",
             label="n = {0}, similarity_step = {1}".format(str(n), str(similarity_step)))
    plt.plot(similarity_scores, [mean_score for _ in similarity_scores], label="mean bluert score")
    plt.title(
        "Average Similarity Between Test QnA and 'n' most similat Training QnAs vs BLEURT Score of Generated Explanation for that test QnA")
    plt.xlabel(
        "average similarity between test Q/A and 'n' most similar Training QnAs - " + similarity_measure.__name__)
    plt.ylabel("(average) bleurt score of generated sentence")
    plt.legend(loc="upper left")
    figure.show()


# how faithful is the model to the input (retrieved) facts?
def no_generated_explanations_vs_no_explanations_copied_from_input(questions_and_answers_with_seperator,
                                                                   generated_explanations_with_separator):
    figure = get_figure()

    no_gen_to_no_copied = {}
    no_gen_to_no_copied_no_rep = {}

    for i in range(len(generated_explanations_with_separator)):
        input_facts = questions_and_answers_with_seperator[i].split("@@")[1].split("££")
        gen_facts = generated_explanations_with_separator[i].split("$$")

        no_gen_facts = len(gen_facts)

        input_facts_bows = [get_bow_of_fact(input_fact) for input_fact in input_facts]
        gen_facts_bows = [get_bow_of_fact(gen_fact) for gen_fact in gen_facts]

        no_copied_facts = 0
        no_copied_facts_no_rep = 0

        for input_fact_bow in input_facts_bows:
            count = gen_facts_bows.count(input_fact_bow)
            if count > 0:
                no_copied_facts_no_rep += 1
                no_copied_facts += count

        if no_gen_facts in no_gen_to_no_copied:
            no_gen_to_no_copied[no_gen_facts].append(no_copied_facts)
            no_gen_to_no_copied_no_rep[no_gen_facts].append(no_copied_facts_no_rep)

        else:
            no_gen_to_no_copied[no_gen_facts] = [no_copied_facts]
            no_gen_to_no_copied_no_rep[no_gen_facts] = [no_copied_facts_no_rep]

    no_gen = sorted(list(no_gen_to_no_copied.keys()))

    # get mean no. copied facts
    no_copied_all = []
    no_copied_all_no_rep = []
    for k in no_gen:
        no_copied_all += no_gen_to_no_copied[k]
        no_copied_all_no_rep += no_gen_to_no_copied_no_rep[k]

    mean_no_copied = np.mean(no_copied_all)
    mean_no_copied_no_rep = np.mean(no_copied_all_no_rep)

    # get means for each no of generated facts
    no_copied_means = [np.mean(no_gen_to_no_copied[k]) for k in no_gen]
    no_copied_means_no_rep = [np.mean(no_gen_to_no_copied_no_rep[k]) for k in no_gen]

    # plt.plot(no_gen, no_copied_means, marker="o", markersize=5, linestyle="dashed",
    #          label="allow repeatedly copied input facts, mean = {0}".format(mean_no_copied))
    plt.plot(no_gen, no_copied_means_no_rep, marker="o", markersize=5, linestyle="dashed",
             label="don't allow repeatedly copied input facts, mean = {0}".format(mean_no_copied_no_rep))

    plt.title("No. Generated facts vs No facts Copied from retrieved facts in input")
    plt.xlabel("No. Generated facts")
    plt.ylabel("No copied facts from input")
    plt.legend(loc="upper left")

    figure.show()


def get_no_retrieved_samples_copied_to_output(generated_with_seperater, retrieved_with_separator, gen_facts_sep="$$", ret_facts_sep="££"):
    assert len(generated_with_seperater) == len(retrieved_with_separator)

    no_copied_arr = []
    no_copied_no_rep_arr = []

    for i in range(len(retrieved_with_separator)):
        input_facts = retrieved_with_separator[i].split(ret_facts_sep)
        gen_facts = generated_with_seperater[i].split(gen_facts_sep)

        input_facts_bows = [get_bow_of_fact(input_fact) for input_fact in input_facts]
        gen_facts_bows = [get_bow_of_fact(gen_fact) for gen_fact in gen_facts]

        no_copied_facts = 0
        no_copied_facts_no_rep = 0

        for input_fact_bow in input_facts_bows:
            count = gen_facts_bows.count(input_fact_bow)
            if count > 0:
                no_copied_facts_no_rep += 1
                no_copied_facts += count

        no_copied_arr.append(no_copied_facts)
        no_copied_no_rep_arr.append(no_copied_facts_no_rep)

    return no_copied_arr, no_copied_no_rep_arr

# how faithful should the model ideally be? not really insightful since it will tend to copy more
# when it needs to generate more facts, and when it generates more facts score tends to be less
def no_facts_copied_from_input_vs_score(questions_and_answers_with_seperator, generated_explanations_with_separator,
                                        scores):
    figure = get_figure()

    no_copied_to_score = {}

    for i in range(len(questions_and_answers_with_seperator)):
        input_facts = questions_and_answers_with_seperator[i].split("@@")[1].split("££")
        gen_facts = generated_explanations_with_separator[i].split("$$")

        input_facts_bows = [get_bow_of_fact(input_fact) for input_fact in input_facts]
        gen_facts_bows = [get_bow_of_fact(gen_fact) for gen_fact in gen_facts]

        no_copied_facts = 0
        for gen_fact_bow in gen_facts_bows:
            if gen_fact_bow in input_facts_bows:
                no_copied_facts += 1

        if no_copied_facts in no_copied_to_score:
            no_copied_to_score[no_copied_facts].append(scores[i])
        else:
            no_copied_to_score[no_copied_facts] = [scores[i]]

    no_copied = sorted(list(no_copied_to_score.keys()))
    mean_scores = [np.mean(no_copied_to_score[k]) for k in no_copied]

    plt.plot(no_copied, mean_scores, marker="o", markersize=5, linestyle="dashed")
    plt.title("No. copied facts from input vs score")
    plt.xlabel("No. copied facts")
    plt.ylabel("bleurt score")
    figure.show()


def get_generated_no_exact_repeated_facts(generated_text_with_separators):
    generated_no_exact_rep = []
    no_repeated_facts = 0
    no_generated_facts = 0
    for generated_exp in generated_text_with_separators:
        facts_no_rep = []
        facts = generated_exp.split("$$")
        for fact in facts:
            fact = fact.strip()
            if fact != "" and not fact.isspace():
                if fact not in facts_no_rep:
                    facts_no_rep.append(fact)
                else:
                    no_repeated_facts += 1
                no_generated_facts += 1
        generated_no_exact_rep.append(".".join(facts_no_rep))
    return generated_no_exact_rep, no_repeated_facts / no_generated_facts


def preprocess_predictions_df(df):
    generated_text = []
    generated_text_with_no_exact_repetitions = []
    reference_text = []
    no_explanations_reference = []
    no_explanations_generated = []

    reference_text_with_separator = []
    generated_text_with_separator = []

    questions_and_answers = []
    questions_and_answers_with_separator = []

    # with separator
    for x in df["Generated Text"]:
        if END_SEP in x:
            x = x[:x.index(END_SEP)].strip()
        generated_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace(
                "<|endoftext|>", "").replace("%%", "$$").replace("&&", "$$").replace("||", "$$").replace(";", " "))
    for x in df["Actual Text"]:
        if END_SEP in x:
            x = x[:x.index(END_SEP)].strip()
        reference_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace(
                "<|endoftext|>", "").replace("%%", "$$").replace("&&", "$$").replace("||", "$$").replace(";", " "))

    for x in df["Questions"]:
        questions_and_answers_with_separator.append(x)
        if "@@" in x:
            x = x.split("@@")[0]
        questions_and_answers.append(x.replace("<|endoftext|>", ""))

    # without separator
    for x in generated_text_with_separator:
        no_explanations_generated.append(x.count("$$") + x.count("%%") + x.count("&&") + x.count("||") + 1)
        generated_text.append(x.replace("$$", ".").replace("%%", ".").replace("&&", ".").replace("||", "."))
    for x in reference_text_with_separator:
        no_explanations_reference.append(x.count("$$") + x.count("%%") + x.count("&&") + x.count("||") + 1)
        reference_text.append(x.replace("$$", ".").replace("%%", ".").replace("&&", ".").replace("||", "."))

    generated_text_with_no_exact_repetitions, no_repeated_to_no_generated_ratio = get_generated_no_exact_repeated_facts(
        generated_text_with_separator)

    return (questions_and_answers, questions_and_answers_with_separator,
            reference_text, reference_text_with_separator,
            generated_text, generated_text_with_separator,
            generated_text_with_no_exact_repetitions, no_repeated_to_no_generated_ratio,
            no_explanations_reference, no_explanations_generated)


if __name__ == "__main__":


    df_predictions = pd.read_csv(DEV_PREDICTIONS_CSV_PATH, delimiter=",")

    (questions_and_answers, questions_and_answers_with_separator,
     reference_text, reference_text_with_separator,
     generated_text, generated_text_with_separator,
     generated_text_with_no_exact_repetitions, no_repeated_to_no_generated_ratio,
     no_explanations_reference, no_explanations_generated) = preprocess_predictions_df(df_predictions)

    try:
        bleurt_scores = df_predictions[BLEURT_SCORES]
    except KeyError:
        bleurt_scores, bleurt_mean_score, bleurt_best_df, bleurt_worst_df = evaluate(metric_key="bleurt",
                                                                                     generated=generated_text_with_no_exact_repetitions,
                                                                                     references=reference_text,
                                                                                     questions=df_predictions[
                                                                                         "Questions"])
        # save new csv with scores
        df_predictions["bleurt_scores"] = bleurt_scores
        df_predictions.to_csv(DEV_PREDICTIONS_CSV_PATH.replace(".csv", "_no_rep_with_bleurt_scores.csv"))

    no_facts_in_reference_vs_no_facts_in_generated(reference_text_with_separator, generated_text_with_separator,
                                                   show_total_no_generated_facts=True,
                                                   show_no_generated_facts_in_reference=True,
                                                   show_no_repeatedly_generated_facts=True)

    show_plots()
    sys.exit()

    # shows how the score decreases as the explanation contains more hops
    no_hops_in_reference_vs_score(no_hops_reference=no_explanations_reference,
                                  scores=bleurt_scores)

    # from the facts that the model generates: how many are relevant? and how many are repeated?
    no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator=reference_text_with_separator,
                                                                generated_with_separator=generated_text_with_separator)

    # expectation is that the more similar the QnA is to the golden explanation, the better the model will do
    similarity_score_for_QnA_and_reference_vs_BLEURT_score_of_generated_explanation(
        questions_and_answers=questions_and_answers,
        scores=df_predictions[BLEURT_SCORES],
        references=reference_text,
        similarity_measure=jaccard_similarity,
        similarity_step=10
    )

    # see if model can generalize to OOD questions
    # expectation: the more similar test question is to train questions the better the model will do
    average_similarity_between_test_and_train_samples_vs_bluert_score(
        qna_testing_set=questions_and_answers,
        qna_training_set=pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")["question_and_answer"],
        similarity_measure=jaccard_similarity,
        scores=df_predictions[BLEURT_SCORES],
        n=3,
        similarity_step=10
    )

    # ideally this should be y = x, see if the model generates enough facts
    no_explanations_in_reference_vs_no_explanations_in_generated(no_explanations_reference=no_explanations_reference,
                                                                 no_explanations_generated=no_explanations_generated)

    # todo change
    questions_and_answers_with_separator = \
    pd.read_csv("evaluation/BART-retrieve-prompt/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")[
        "Questions"]
    no_generated_explanations_vs_no_explanations_copied_from_input(
        questions_and_answers_with_seperator=questions_and_answers_with_separator,
        generated_explanations_with_separator=generated_text_with_separator)

    no_facts_copied_from_input_vs_score(questions_and_answers_with_seperator=questions_and_answers_with_separator,
                                        generated_explanations_with_separator=generated_text_with_separator,
                                        scores=bleurt_scores)

    # todo: fix
    # does the model's performance degrade when the questions are longer?
    # no_words_in_question_vs_score(questions=questions_and_answers,
    #                              scores=bleurt_scores)

    show_plots()
    sys.exit()
