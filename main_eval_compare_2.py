import sys
import heapq
from typing import List

import nltk
nltk.download('stopwords')
#nltk.download("punc")
import pandas as pd
import numpy as np
from datasets import load_metric
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



###############################################
## todo: change file path
###############################################
TEST_PREDICTIONS_CSV_PATH_1 = "evaluation/BART-plain/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv"  # "evaluation/t5-plain/validation_predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting.csv"  #"evaluation/BART-lr-3e-5/test_predictions_vs_actuals_with_BLEURT_scores.csv"  # "outputs/dummy_predicions_with_BLEURT_scores.csv"  # "./evaluation/predictions_vs_actuals-t5-plain-from-QnA-with-data-splitting.csv" #"./evaluation/predictions_vs_actuals-t5-plain-from-hypothesis-with-data-splitting.csv"
TEST_PREDICTIONS_CSV_PATH_2 = "evaluation/BART-retrieve-prompt/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv"
MODEL1 = "BART plain"
MODEL2 = "BART retrieve prompt"

TRAINING_DATA_CSV_PATH = "data/v2-proper-data/train_data_wed.csv"

STOP_WORDS = stopwords.words("english")

###############################################
BLERT_SCORES = "bleurt_scores"
FIGURE_COUNTER = 0

stemmer = PorterStemmer()



def show_plots():
    plt.show()


def get_figure():
    global FIGURE_COUNTER
    f = plt.figure(FIGURE_COUNTER)
    FIGURE_COUNTER += 1
    return f


def no_hops_in_reference_vs_score(no_hops_reference, scores1, scores2):
    mean_score1 = np.mean(scores1)
    no_hops_to_scores1 = {}
    for i in range(len(no_hops_reference)):
        if no_hops_reference[i] in no_hops_to_scores1:
            no_hops_to_scores1[no_hops_reference[i]].append(scores1[i])
        else:
            no_hops_to_scores1[no_hops_reference[i]] = [scores1[i]]

    hop_numbers = sorted(list(no_hops_to_scores1.keys()))
    average_scores1 = [np.mean(no_hops_to_scores1[hop_no]) for hop_no in hop_numbers]

    mean_score2 = np.mean(scores2)
    no_hops_to_scores2 = {}
    for i in range(len(no_hops_reference)):
        if no_hops_reference[i] in no_hops_to_scores2:
            no_hops_to_scores2[no_hops_reference[i]].append(scores2[i])
        else:
            no_hops_to_scores2[no_hops_reference[i]] = [scores2[i]]

    hop_numbers = sorted(list(no_hops_to_scores2.keys()))
    average_scores2 = [np.mean(no_hops_to_scores2[hop_no]) for hop_no in hop_numbers]

    figure = get_figure()
    plt.plot(hop_numbers, average_scores1, marker="o", linestyle="dashed", label=MODEL1)
    plt.plot(hop_numbers, average_scores2, marker="o", linestyle="dashed", label=MODEL2)

    # add line at mean
    plt.plot(hop_numbers, [mean_score1] * len(hop_numbers), label=MODEL1 + " mean")
    plt.plot(hop_numbers, [mean_score2] * len(hop_numbers), label=MODEL2 + " mean")

    plt.title("No. Explanations in reference vs Score")
    plt.xlabel("no. explanations in ref")
    plt.ylabel("bleurt scores")
    plt.legend(loc="upper right")

    figure.show()


# TODO: SMOOTH
def no_words_in_question_vs_score(questions, scores):
    mean_score = np.mean(scores)
    no_words_in_questions = []
    for question in questions:
        no_words_in_questions.append(len(question.split()))

    no_words_in_questions_to_scores = {}
    for i in range(len(no_words_in_questions)):
        if no_words_in_questions[i] in no_words_in_questions_to_scores:
            no_words_in_questions_to_scores[no_words_in_questions[i]].append(scores[i])
        else:
            no_words_in_questions_to_scores[no_words_in_questions[i]] = [scores[i]]

    no_words_in_questions = sorted(list(no_words_in_questions_to_scores.keys()))

    scores = [np.mean(no_words_in_questions_to_scores[no_words_in_question]) for no_words_in_question in
              no_words_in_questions]

    figure = get_figure()

    plt.plot(no_words_in_questions, scores, marker="o", linestyle="dashed")
    plt.plot(no_words_in_questions, [mean_score] * len(no_words_in_questions))
    plt.title("No. words in question vs score")
    plt.xlabel("no. words in question")
    plt.ylabel("bleurt score")
    figure.show()


def no_explanations_in_reference_vs_no_explanations_in_generated(no_explanations_reference, no_explanations_generated1, no_explanations_generated2):
    no_exp_ref_to_no_exp_gen1 = {}
    no_exp_ref_to_no_exp_gen2 = {}

    for i in range(len(no_explanations_reference)):
        if no_explanations_reference[i] in no_exp_ref_to_no_exp_gen1:
            no_exp_ref_to_no_exp_gen1[no_explanations_reference[i]].append(no_explanations_generated1[i])
        else:
            no_exp_ref_to_no_exp_gen1[no_explanations_reference[i]] = [no_explanations_generated1[i]]

    for i in range(len(no_explanations_reference)):
        if no_explanations_reference[i] in no_exp_ref_to_no_exp_gen2:
            no_exp_ref_to_no_exp_gen2[no_explanations_reference[i]].append(no_explanations_generated2[i])
        else:
            no_exp_ref_to_no_exp_gen2[no_explanations_reference[i]] = [no_explanations_generated2[i]]


    no_exp_refs = sorted(list(no_exp_ref_to_no_exp_gen1.keys()))
    no_exp_gens1 = [np.mean(no_exp_ref_to_no_exp_gen1[no_exp_ref]) for no_exp_ref in no_exp_refs]
    no_exp_gens2 = [np.mean(no_exp_ref_to_no_exp_gen2[no_exp_ref]) for no_exp_ref in no_exp_refs]

    figure = get_figure()

    plt.plot(no_exp_refs, no_exp_gens1, marker="o", linestyle="dashed", label=MODEL1)
    plt.plot(no_exp_refs, no_exp_gens2, marker="o", linestyle="dashed", label=MODEL2)

    plt.plot(no_exp_refs, no_exp_refs, label="ideal situation")
    plt.title("No. explanations in reference vs No. explanations in generated")
    plt.xlabel("no. explanations in ref")
    plt.ylabel("no. explanations in gen")
    plt.legend(loc="upper left")

    figure.show()


def get_bow_of_fact(fact):
    return set(
        stemmer.stem(word.lower().strip()) for word in fact.split() if
        #word.lower().strip() for word in fact.split() if
        word.lower().strip() not in STOP_WORDS and word != "" and not word.isspace()
    )


# 2 facts are the same if their BOWs without stopwords are the same
def no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator, generated_with_separator1, generated_with_separator2):
    # repeated facts are only counted once
    NO_FACTS_OCCURRING_IN_REF = "no_facts_occurring_in_reference"
    NO_REPEATED_FACTS = "no_repeated_facts"
    MEAN_NO_FACTS_OCCURRING_IN_REF = "mean_no_facts_occurring_in_reference"
    MEAN_NO_REPEATED_FACTS = "mean_no_repeated_facts"

    no_gen_facts_to_stats1 = {}
    no_gen_facts_to_stats2 = {}

    for i in range(len(references_with_seperator)):
        reference_facts = references_with_seperator[i].lower().replace(";", " ").split("$$")
        reference_facts[:] = [ref_fact.strip() for ref_fact in reference_facts]
        reference_facts_bows = [get_bow_of_fact(ref_fact) for ref_fact in reference_facts]

        generated_facts1 = generated_with_separator1[i].lower().replace(";", " ").split("$$")
        generated_facts1[:] = [gen_fact.strip() for gen_fact in generated_facts1]
        generated_facts_bows1 = [get_bow_of_fact(gen_fact) for gen_fact in generated_facts1]

        generated_facts2 = generated_with_separator2[i].lower().replace(";", " ").split("$$")
        generated_facts2[:] = [gen_fact.strip() for gen_fact in generated_facts2]
        generated_facts_bows2 = [get_bow_of_fact(gen_fact) for gen_fact in generated_facts2]

        # calculate number of repeated generated facts
        unique_gen_facts_bows1 = []
        for gen_fact_bow in generated_facts_bows1:
            if gen_fact_bow not in unique_gen_facts_bows1:
                unique_gen_facts_bows1.append(gen_fact_bow)

        no_gen_facts_in_ref1 = 0
        for gen_fact_bow in unique_gen_facts_bows1:
            if gen_fact_bow in reference_facts_bows:
                no_gen_facts_in_ref1 += 1

        no_repeated_gen_facts1 = len(generated_facts_bows1) - len(unique_gen_facts_bows1)
        no_generated_facts1 = len(generated_facts1)

        unique_gen_facts_bows2 = []
        for gen_fact_bow in generated_facts_bows2:
            if gen_fact_bow not in unique_gen_facts_bows2:
                unique_gen_facts_bows2.append(gen_fact_bow)

        no_gen_facts_in_ref2 = 0
        for gen_fact_bow in unique_gen_facts_bows2:
            if gen_fact_bow in reference_facts_bows:
                no_gen_facts_in_ref2 += 1

        no_repeated_gen_facts2 = len(generated_facts_bows2) - len(unique_gen_facts_bows2)
        no_generated_facts2 = len(generated_facts2)


        if no_generated_facts1 in no_gen_facts_to_stats1:
            no_gen_facts_to_stats1[no_generated_facts1][NO_FACTS_OCCURRING_IN_REF].append(no_gen_facts_in_ref1)
            no_gen_facts_to_stats1[no_generated_facts1][NO_REPEATED_FACTS].append(no_repeated_gen_facts1)
        else:
            no_gen_facts_to_stats1[no_generated_facts1] = {
                NO_FACTS_OCCURRING_IN_REF: [no_gen_facts_in_ref1],
                NO_REPEATED_FACTS: [no_repeated_gen_facts1]
            }

        if no_generated_facts2 in no_gen_facts_to_stats2:
            no_gen_facts_to_stats2[no_generated_facts2][NO_FACTS_OCCURRING_IN_REF].append(no_gen_facts_in_ref2)
            no_gen_facts_to_stats2[no_generated_facts2][NO_REPEATED_FACTS].append(no_repeated_gen_facts2)
        else:
            no_gen_facts_to_stats2[no_generated_facts2] = {
                NO_FACTS_OCCURRING_IN_REF: [no_gen_facts_in_ref2],
                NO_REPEATED_FACTS: [no_repeated_gen_facts2]
            }

    for k in no_gen_facts_to_stats1.keys():
        no_gen_facts_to_stats1[k][MEAN_NO_FACTS_OCCURRING_IN_REF] = np.mean(
            no_gen_facts_to_stats1[k][NO_FACTS_OCCURRING_IN_REF])
        no_gen_facts_to_stats1[k][MEAN_NO_REPEATED_FACTS] = np.mean(no_gen_facts_to_stats1[k][NO_REPEATED_FACTS])

    for k in no_gen_facts_to_stats2.keys():
        no_gen_facts_to_stats2[k][MEAN_NO_FACTS_OCCURRING_IN_REF] = np.mean(
            no_gen_facts_to_stats2[k][NO_FACTS_OCCURRING_IN_REF])
        no_gen_facts_to_stats2[k][MEAN_NO_REPEATED_FACTS] = np.mean(no_gen_facts_to_stats2[k][NO_REPEATED_FACTS])


    generated_facts_numbers1 = sorted(list(no_gen_facts_to_stats1.keys()))
    no_gen_facts_occurring_in_ref_mean1 = [no_gen_facts_to_stats1[n][MEAN_NO_FACTS_OCCURRING_IN_REF] for n in
                                          generated_facts_numbers1]
    no_repeated_gen_facts1 = [no_gen_facts_to_stats1[n][MEAN_NO_REPEATED_FACTS] for n in generated_facts_numbers1]

    generated_facts_numbers2 = sorted(list(no_gen_facts_to_stats2.keys()))
    no_gen_facts_occurring_in_ref_mean2 = [no_gen_facts_to_stats2[n][MEAN_NO_FACTS_OCCURRING_IN_REF] for n in
                                          generated_facts_numbers2]
    no_repeated_gen_facts2 = [no_gen_facts_to_stats2[n][MEAN_NO_REPEATED_FACTS] for n in generated_facts_numbers2]

    figure = get_figure()

    plt.plot(generated_facts_numbers1, no_gen_facts_occurring_in_ref_mean1, marker="o", linestyle="dashed",
             label=MODEL1)
    plt.plot(generated_facts_numbers2, no_gen_facts_occurring_in_ref_mean2, marker="o", linestyle="solid", label=MODEL2)
    plt.legend(loc="upper left")
    plt.title("No. Generated Facts vs No. Generated Facts Occurring in Reference")
    plt.xlabel("no. generated facts")
    plt.ylabel("no. generated facts occurring in reference")

    figure.show()



# the higher the score the more similar s1 and s2
def jaccard_similarity(s1, s2):
    """
    J(A, B) = |A intersection B| / (|A| + |B| - |A intersection B|)
    """
    s1 = s1.replace(";", " ").replace(".", " ")
    s2 = s2.replace(";", " ").replace(".", " ")

    s1_bow = get_bow_of_fact(s1)
    s2_bow = get_bow_of_fact(s2)

    s1_bow_inter_s2_bow = s1_bow.intersection(s2_bow)
    score = len(s1_bow_inter_s2_bow) / (len(s1_bow) + len(s2_bow) - len(s1_bow_inter_s2_bow))
    return score * 100


# TODO: need a new way of passing questions_and_answers with retrieval strategy, pass original df before modificaiton
def similarity_score_for_QnA_and_reference_vs_BLEURT_score_of_generated_explanation(questions_and_answers, references,
                                                                                    scores1, scores2,
                                                                                    similarity_measure,
                                                                                    similarity_step):
    assert len(questions_and_answers) == len(references) == len(scores1) == len(scores2)
    mean_score1 = np.mean(scores1)
    mean_score2 = np.mean(scores2)

    similarity_to_score1 = {}
    similarity_to_score2 = {}

    similarities_to_show = np.array([i for i in range(0, 100, similarity_step)])
    for similarity_to_show in similarities_to_show:
        similarity_to_score1[similarity_to_show] = []
        similarity_to_score2[similarity_to_show] = []

    for i in range(len(questions_and_answers)):
        similarity = similarity_measure(questions_and_answers[i], references[i])

        similarity_diff = abs(similarity - similarities_to_show)
        closest_similarity_index = np.argmin(similarity_diff)
        closest_similarity = similarities_to_show[closest_similarity_index]

        similarity_to_score1[closest_similarity].append(scores1[i])
        similarity_to_score2[closest_similarity].append(scores2[i])


    # remove similarities that have no scores associated to them
    similarities = sorted(sim for sim in similarity_to_score1.keys() if similarity_to_score1[sim] != [])
    scores1 = [np.mean(similarity_to_score1[s]) for s in similarities]
    scores2 = [np.mean(similarity_to_score2[s]) for s in similarities]

    figure = get_figure()

    plt.plot(similarities, scores1, marker="o", linestyle="dashed", label=MODEL1)
    plt.plot(similarities, scores2, marker="o", linestyle="dashed", label=MODEL2)

    plt.plot(similarities, [mean_score1 for _ in similarities], label=MODEL1 + " mean")
    plt.plot(similarities, [mean_score2 for _ in similarities], label=MODEL2 + " mean")

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


def average_similarity_between_test_and_train_samples_vs_bluert_score(qna_testing_set, qna_training_set, scores1, scores2, n,
                                                                      similarity_measure, similarity_step):
    mean_score1 = np.mean(scores1)
    mean_score2 = np.mean(scores2)

    average_similarities_to_bleu_scores1 = {}
    average_similarities_to_bleu_scores2 = {}

    similarities_to_show = np.array([i for i in range(0, 100, similarity_step)])
    for similarity_to_show in similarities_to_show:
        average_similarities_to_bleu_scores1[similarity_to_show] = []
        average_similarities_to_bleu_scores2[similarity_to_show] = []


    for i in range(len(qna_testing_set)):
        similarities = []
        for training_qna in qna_training_set:
            similarities.append(similarity_measure(qna_testing_set[i], training_qna))
        average_n_most_similar = np.mean(heapq.nlargest(n, similarities))

        similarity_diff = abs(average_n_most_similar - similarities_to_show)
        closest_similarity_index = np.argmin(similarity_diff)
        closest_similarity = similarities_to_show[closest_similarity_index]
        average_similarities_to_bleu_scores1[closest_similarity].append(scores1[i])
        average_similarities_to_bleu_scores2[closest_similarity].append(scores2[i])


    figure = get_figure()

    similarity_scores = sorted(
        sim for sim in average_similarities_to_bleu_scores1.keys() if average_similarities_to_bleu_scores1[sim] != []
            and average_similarities_to_bleu_scores2[sim] != [])

    bluert_scores1 = [np.mean(average_similarities_to_bleu_scores1[similarity_score]) for similarity_score in
                     similarity_scores]
    bluert_scores2 = [np.mean(average_similarities_to_bleu_scores2[similarity_score]) for similarity_score in
                     similarity_scores]

    plt.plot(similarity_scores, bluert_scores1, marker="o", markersize=5, linestyle="solid",
             label=MODEL1)
    plt.plot(similarity_scores, bluert_scores2, marker="o", markersize=5, linestyle="solid",
             label=MODEL2)

    plt.plot(similarity_scores, [mean_score1 for _ in similarity_scores], label=MODEL1 + " - mean bluert score")
    plt.plot(similarity_scores, [mean_score2 for _ in similarity_scores], label=MODEL2 + " - mean bluert score")

    plt.title(
        "Average Similarity Between Test QnA and '{0}' most similar Training QnAs vs BLEURT Score of Generated Explanation for that test QnA".format(n))
    plt.xlabel(
        "average similarity between test Q/A and '{0}' most similar Training QnAs - ".format(n) + similarity_measure.__name__)
    plt.ylabel("(average) bleurt score of generated sentence")
    plt.legend(loc="upper left")
    figure.show()


# how faithful is the model to the input (retrieved) facts?
def no_generated_explanations_vs_no_explanations_copied_from_input(questions_and_answers_with_seperator,
                                                                   generated_explanations_with_separator1,
                                                                   generated_explanations_with_separator2):
    figure = get_figure()

    no_gen_to_no_copied1 = {}
    no_gen_to_no_copied_no_rep1 = {}

    no_gen_to_no_copied2 = {}
    no_gen_to_no_copied_no_rep2 = {}

    for i in range(len(generated_explanations_with_separator1)):
        input_facts = questions_and_answers_with_seperator[i].split("@@")[1].split("££")
        gen_facts1 = generated_explanations_with_separator1[i].split("$$")
        gen_facts2 = generated_explanations_with_separator2[i].split("$$")

        no_gen_facts1 = len(gen_facts1)
        no_gen_facts2 = len(gen_facts2)

        input_facts_bows = [get_bow_of_fact(input_fact) for input_fact in input_facts]
        gen_facts_bows1 = [get_bow_of_fact(gen_fact) for gen_fact in gen_facts1]
        gen_facts_bows2 = [get_bow_of_fact(gen_fact) for gen_fact in gen_facts2]


        no_copied_facts1 = 0
        no_copied_facts_no_rep1 = 0

        no_copied_facts2 = 0
        no_copied_facts_no_rep2 = 0

        for input_fact_bow in input_facts_bows:
            count1 = gen_facts_bows1.count(input_fact_bow)
            count2 = gen_facts_bows2.count(input_fact_bow)
            if count1 > 0:
                no_copied_facts_no_rep1 += 1
                no_copied_facts1 += count1
            if count2 > 0:
                no_copied_facts_no_rep2 += 1
                no_copied_facts2 += count2

        if no_gen_facts1 in no_gen_to_no_copied1:
            no_gen_to_no_copied1[no_gen_facts1].append(no_copied_facts1)
            no_gen_to_no_copied_no_rep1[no_gen_facts1].append(no_copied_facts_no_rep1)

        else:
            no_gen_to_no_copied1[no_gen_facts1] = [no_copied_facts1]
            no_gen_to_no_copied_no_rep1[no_gen_facts1] = [no_copied_facts_no_rep1]

        if no_gen_facts2 in no_gen_to_no_copied2:
            no_gen_to_no_copied2[no_gen_facts2].append(no_copied_facts2)
            no_gen_to_no_copied_no_rep2[no_gen_facts2].append(no_copied_facts_no_rep2)

        else:
            no_gen_to_no_copied2[no_gen_facts2] = [no_copied_facts2]
            no_gen_to_no_copied_no_rep2[no_gen_facts2] = [no_copied_facts_no_rep2]

    no_gen1 = sorted(list(no_gen_to_no_copied1.keys()))
    no_gen2 = sorted(list(no_gen_to_no_copied2.keys()))

    # get mean no. copied facts
    no_copied_all1 = []
    no_copied_all_no_rep1 = []
    for k in no_gen1:
        no_copied_all1 += no_gen_to_no_copied1[k]
        no_copied_all_no_rep1 += no_gen_to_no_copied_no_rep1[k]

    no_copied_all2 = []
    no_copied_all_no_rep2 = []
    for k in no_gen2:
        no_copied_all2 += no_gen_to_no_copied2[k]
        no_copied_all_no_rep2 += no_gen_to_no_copied_no_rep2[k]

    mean_no_copied1 = np.mean(no_copied_all1)
    mean_no_copied_no_rep1 = np.mean(no_copied_all_no_rep1)

    # get means for each no of generated facts
    no_copied_means1 = [np.mean(no_gen_to_no_copied1[k]) for k in no_gen1]
    no_copied_means_no_rep1 = [np.mean(no_gen_to_no_copied_no_rep1[k]) for k in no_gen1]

    mean_no_copied2 = np.mean(no_copied_all2)
    mean_no_copied_no_rep2 = np.mean(no_copied_all_no_rep2)

    # get means for each no of generated facts
    no_copied_means2 = [np.mean(no_gen_to_no_copied2[k]) for k in no_gen2]
    no_copied_means_no_rep2 = [np.mean(no_gen_to_no_copied_no_rep2[k]) for k in no_gen2]

    # plt.plot(no_gen, no_copied_means, marker="o", markersize=5, linestyle="dashed",
    #          label="allow repeatedly copied input facts, mean = {0}".format(mean_no_copied))
    plt.plot(no_gen1, no_copied_means_no_rep1, marker="o", markersize=5, linestyle="dashed",
             label=MODEL1 + ", mean = {0}".format(mean_no_copied_no_rep1))
    plt.plot(no_gen2, no_copied_means_no_rep2, marker="o", markersize=5, linestyle="dashed",
             label=MODEL2 + ", mean = {0}".format(mean_no_copied_no_rep2))

    plt.title("No. Generated facts vs No facts Copied from retrieved facts in input")
    plt.xlabel("No. Generated facts")
    plt.ylabel("No copied facts from input")
    plt.legend(loc="upper left")

    figure.show()



def get_generated_no_exact_repeated_facts(generated_text_with_separators):
    generated_no_exact_rep = []
    no_repeated_facts = 0
    no_generated_facts = 0
    for generated_exp in generated_text_with_separators:
        facts_no_rep = []
        facts = generated_exp.split("$$")
        for fact in facts:
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
        generated_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace(
                "<|endoftext|>", ""))
    for x in df["Actual Text"]:
        reference_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace(
                "<|endoftext|>", ""))
    for x in df["Questions"]:
        questions_and_answers_with_separator.append(x)
        if "@@" in x:
            x = x.split("@@")[0]
        questions_and_answers.append(x.replace("<|endoftext|>", ""))

    # without separator
    for x in generated_text_with_separator:
        no_explanations_generated.append(x.count("$$") + 1)
        generated_text.append(x.replace("$$", "."))
    for x in reference_text_with_separator:
        no_explanations_reference.append(x.count("$$") + 1)
        reference_text.append(x.replace("$$", "."))

    generated_text_with_no_exact_repetitions, no_repeated_to_no_generated_ratio = get_generated_no_exact_repeated_facts(
        generated_text_with_separator)

    return (questions_and_answers, questions_and_answers_with_separator,
            reference_text, reference_text_with_separator,
            generated_text, generated_text_with_separator,
            generated_text_with_no_exact_repetitions, no_repeated_to_no_generated_ratio,
            no_explanations_reference, no_explanations_generated)


if __name__ == "__main__":
    df_predictions1 = pd.read_csv(TEST_PREDICTIONS_CSV_PATH_1, delimiter=",")

    (questions_and_answers1, questions_and_answers_with_separator1,
     reference_text1, reference_text_with_separator1,
     generated_text1, generated_text_with_separator1,
     generated_text_with_no_exact_repetitions1, no_repeated_to_no_generated_ratio1,
     no_explanations_reference1, no_explanations_generated1) = preprocess_predictions_df(df_predictions1)
    bleurt_scores1 = df_predictions1[BLERT_SCORES]

    df_predictions2 = pd.read_csv(TEST_PREDICTIONS_CSV_PATH_2, delimiter=",")

    (questions_and_answers2, questions_and_answers_with_separator2,
     reference_text2, reference_text_with_separator2,
     generated_text2, generated_text_with_separator2,
     generated_text_with_no_exact_repetitions2, no_repeated_to_no_generated_ratio2,
     no_explanations_reference2, no_explanations_generated2) = preprocess_predictions_df(df_predictions2)
    bleurt_scores2 = df_predictions2[BLERT_SCORES]

    ## plots

    # shows how the score decreases as the explanation contains more hops
    # no_hops_in_reference_vs_score(no_hops_reference=no_explanations_reference1,
    #                               scores1=bleurt_scores1,
    #                               scores2=bleurt_scores2)

    # # from the facts that the model generates: how many are relevant? and how many are repeated?
    # no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator=reference_text_with_separator1,
    #                                                             generated_with_separator1=generated_text_with_separator1,
    #                                                             generated_with_separator2=generated_text_with_separator2)
    #
    # # expectation is that the more similar the QnA is to the golden explanation, the better the model will do
    # similarity_score_for_QnA_and_reference_vs_BLEURT_score_of_generated_explanation(
    #     questions_and_answers=questions_and_answers1,
    #     scores1=df_predictions1[BLERT_SCORES],
    #     scores2=df_predictions2[BLERT_SCORES],
    #     references=reference_text1,
    #     similarity_measure=jaccard_similarity,
    #     similarity_step=10
    # )
    #
    # # see if model can generalize to OOD questions
    # # expectation: the more similar test question is to train questions the better the model will do
    # average_similarity_between_test_and_train_samples_vs_bluert_score(
    #     qna_testing_set=questions_and_answers1,
    #     qna_training_set=pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")["question_and_answer"],
    #     similarity_measure=jaccard_similarity,
    #     scores1=df_predictions1[BLERT_SCORES],
    #     scores2=df_predictions2[BLERT_SCORES],
    #     n=3,
    #     similarity_step=10
    # )
    #
    # ideally this should be y = x, see if the model generates enough facts
    # no_explanations_in_reference_vs_no_explanations_in_generated(no_explanations_reference=no_explanations_reference1,
    #                                                              no_explanations_generated1=no_explanations_generated1,
    #                                                              no_explanations_generated2=no_explanations_generated2)
    #
    # # todo change
    questions_and_answers_with_separator = pd.read_csv("evaluation/BART-retrieve-prompt/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")["Questions"]
    no_generated_explanations_vs_no_explanations_copied_from_input(
        questions_and_answers_with_seperator=questions_and_answers_with_separator,
        generated_explanations_with_separator1=generated_text_with_separator1,
        generated_explanations_with_separator2=generated_text_with_separator2
    )


    #todo: fix
    #does the model's performance degrade when the questions are longer?
    #no_words_in_question_vs_score(questions=questions_and_answers,
    #                              scores=bleurt_scores)

    show_plots()
    sys.exit()
