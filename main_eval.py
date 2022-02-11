import sys
import heapq
from typing import List

import nltk
import pandas as pd
import numpy as np
from datasets import load_metric
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

###############################################
## todo: change file path
###############################################
DEV_PREDICTIONS_CSV_PATH = "evaluation/t-cvae-1/test_predictions_vs_actuals_with_BLEURT_scores.csv" #"evaluation/T5/validation_predictions_vs_actuals-T5-from-QnA-with-data-splitting.csv"  #"evaluation/BART-lr-3e-5/test_predictions_vs_actuals_with_BLEURT_scores.csv"  # "outputs/dummy_predicions_with_BLEURT_scores.csv"  # "./evaluation/predictions_vs_actuals-T5-from-QnA-with-data-splitting.csv" #"./evaluation/predictions_vs_actuals-T5-from-hypothesis-with-data-splitting.csv"
TRAINING_DATA_CSV_PATH = "data/v2-proper-data/train_data.csv"

num_of_best_worst_explanations = 15
STOP_WORDS = stopwords.words("english")
###############################################
BLERT_SCORES = "bleurt_scores"
FIGURE_COUNTER = 0


def show_plots():
    plt.show()


def get_figure():
    global FIGURE_COUNTER
    f = plt.figure(FIGURE_COUNTER)
    FIGURE_COUNTER += 1
    return f


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


def no_explanations_in_reference_vs_no_explanations_in_generated(no_explanations_reference, no_explanations_generated):
    no_exp_ref_to_no_exp_gen = {}
    for i in range(len(no_explanations_reference)):
        if no_explanations_reference[i] in no_exp_ref_to_no_exp_gen:
            no_exp_ref_to_no_exp_gen[no_explanations_reference[i]].append(no_explanations_generated[i])
        else:
            no_exp_ref_to_no_exp_gen[no_explanations_reference[i]] = [no_explanations_generated[i]]

    no_exp_refs = sorted(list(no_exp_ref_to_no_exp_gen.keys()))
    no_exp_gens = [np.mean(no_exp_ref_to_no_exp_gen[no_exp_ref]) for no_exp_ref in no_exp_refs]

    figure = get_figure()

    plt.plot(no_exp_refs, no_exp_gens, marker="o", linestyle="dashed")
    plt.plot(no_exp_refs, no_exp_refs)
    plt.title("No. explanations in reference vs No. explanations in generated")
    plt.xlabel("no. explanations in ref")
    plt.ylabel("no. explanations in gen")
    figure.show()


def evaluate(metric_key: str, questions, references, generated):
    if metric_key == "bleurt":
        metric = load_metric('bleurt', "bleurt-large-512")
    else:
        raise Exception()

    metric.add_batch(predictions=generated, references=references)
    c = metric.compute()
    scores = np.array(c["scores"])
    scores_mean = np.mean(scores)

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
    print(best_explanations_dict)

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
    print(worst_explanations_dict)
    print("==========================================================")

    return scores, scores_mean, best_explanations_df, worst_explanations_df

# 2 facts are the same if their BOWs without stopwords are the same
def no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator, generated_with_separator):
    # repeated facts are only counted once
    NO_FACTS_OCCURRING_IN_REF = "no_facts_occurring_in_reference"
    NO_REPEATED_FACTS = "no_repeated_facts"
    MEAN_NO_FACTS_OCCURRING_IN_REF = "mean_no_facts_occurring_in_reference"
    MEAN_NO_REPEATED_FACTS = "mean_no_repeated_facts"

    no_gen_facts_to_stats = {}

    for i in range(len(references_with_seperator)):
        reference_facts = references_with_seperator[i].lower().replace(";", " ").split("$$")
        reference_facts[:] = [ref_fact.strip() for ref_fact in reference_facts]
        reference_facts_bows = [set(
            word.strip() for word in ref_fact.split() if word not in STOP_WORDS and word != "" and not word.isspace())
            for ref_fact in
            reference_facts]

        generated_facts = generated_with_separator[i].lower().replace(";", " ").split("$$")
        generated_facts[:] = [gen_fact.strip() for gen_fact in generated_facts]
        generated_facts_bows = [set(
            word.strip() for word in gen_fact.split() if word not in STOP_WORDS and word != "" and not word.isspace())
            for gen_fact in
            generated_facts]

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
    s1 = s1.replace(";", " ").replace(".", " ")
    s2 = s2.replace(";", " ").replace(".", " ")

    s1_bow = set(word.strip() for word in s1.split() if word not in STOP_WORDS and word != "" and not word.isspace())
    s2_bow = set(word.strip() for word in s2.split() if word not in STOP_WORDS and word != "" and not word.isspace())

    s1_bow_inter_s2_bow = s1_bow.intersection(s2_bow)
    score = len(s1_bow_inter_s2_bow) / (len(s1_bow) + len(s2_bow) - len(s1_bow_inter_s2_bow))
    return score * 100


def similarity_score_for_QnA_and_reference_vs_BLEURT_score_of_generated_explanation(questions_and_answers, references, scores,
                                                                                    similarity_measure, similarity_step):
    assert len(questions_and_answers) == len(references) == len(scores)
    mean_score = np.mean(scores)

    similarity_to_score = {}
    similarities_to_show = np.array([i for i in range(0,100,similarity_step)])
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
    plt.xlabel("similarity between Q/A and golden reference - " + str(similarity_measure.__name__) + ", with similarity_step = " + str(similarity_step))
    plt.ylabel("bleurt score of generated sentence")
    plt.legend(loc="upper left")
    figure.show()


def average_similarity_of_each_test_QnA_to_n_closest_QnAs_in_training_set_vs_no_test_samples_with_this_score(qna_testing_set, qna_training_set,
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

            if average_similarity in average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n]:
                average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n][average_similarity] += 1
            else:
                average_similarity_with_n_closest_samples_to_no_of_sentences_with_that_similarity[n][average_similarity] = 1

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


def average_similarity_between_test_and_train_samples_vs_bluert_score(qna_testing_set, qna_training_set, scores, n, similarity_measure, similarity_step):
    mean_score = np.mean(scores)
    average_similarities_to_bleu_scores = {}
    similarities_to_show = np.array([i for i in range(0,100,similarity_step)])
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

    similarity_scores = sorted(sim for sim in average_similarities_to_bleu_scores.keys() if average_similarities_to_bleu_scores[sim] != [])
    bluert_scores = [np.mean(average_similarities_to_bleu_scores[similarity_score]) for similarity_score in similarity_scores]
    plt.plot(similarity_scores, bluert_scores, marker="o", markersize=5, linestyle="solid", label="n = {0}, similarity_step = {1}".format(str(n), str(similarity_step)))
    plt.plot(similarity_scores, [mean_score for _ in similarity_scores], label="mean bluert score")
    plt.title("Average Similarity Between Test QnA and 'n' most similat Training QnAs vs BLEURT Score of Generated Explanation for that test QnA")
    plt.xlabel("average similarity between test Q/A and 'n' most similar Training QnAs - " + similarity_measure.__name__)
    plt.ylabel("(average) bleurt score of generated sentence")
    plt.legend(loc="upper left")
    figure.show()

if __name__ == "__main__":
    df_predictions = pd.read_csv(DEV_PREDICTIONS_CSV_PATH, delimiter=",")

    generated_text = []
    reference_text = []
    no_explanations_reference = []
    no_explanations_generated = []

    reference_text_with_separator = []
    generated_text_with_separator = []

    questions_and_answers = []
    # with separator
    for x in df_predictions["Generated Text"]:
        generated_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace("<|endoftext|>", ""))
    for x in df_predictions["Actual Text"]:
        reference_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace("<|endoftext|>", ""))
    for x in df_predictions["Questions"]:
        questions_and_answers.append(x.replace("<|endoftext|>", ""))

    # without separator
    for x in generated_text_with_separator:
        no_explanations_generated.append(x.count("$$") + 1)
        generated_text.append(x.replace("$$", "."))
    for x in reference_text_with_separator:
        no_explanations_reference.append(x.count("$$") + 1)
        reference_text.append(x.replace("$$", "."))

    # no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator=reference_text_with_separator,
    #                                                             generated_with_separator=generated_text_with_separator)

    # similarity_score_for_QnA_and_reference_vs_BLEURT_score_of_generated_explanation(
    #     questions_and_answers=questions_and_answers,
    #     scores=df_predictions[BLERT_SCORES],
    #     references=reference_text,
    #     similarity_measure=jaccard_similarity,
    #     similarity_step=10
    # )

    #
    average_similarity_between_test_and_train_samples_vs_bluert_score(
        qna_testing_set=questions_and_answers,
        qna_training_set=pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")["question_and_answer"],
        similarity_measure=jaccard_similarity,
        scores=df_predictions[BLERT_SCORES],
        n=3,
        similarity_step=10
    )

    show_plots()
    sys.exit()

    # average_similarity_between_test_and_train_samples_vs_bluert_score(
    #     qna_testing_set=df_predictions["Questions"],
    #     qna_training_set=pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")["question_and_answer"],
    #     similarity_measure=jaccard_similarity,
    #     scores=df_predictions[BLERT_SCORES],
    #     n=1
    # )
    # show_plots()
    # sys.exit()

    # average_similarity_of_each_test_QnA_to_n_closest_QnAs_in_training_set_vs_no_test_samples_with_this_score(
    #     qna_testing_set=df_predictions["Questions"],
    #     qna_training_set=pd.read_csv(TRAINING_DATA_CSV_PATH, "\t")["question_and_answer"],
    #     similarity_measure=jaccard_similarity,
    #     ns=[1, 3, 100]
    # )
    # show_plots()
    # sys.exit()

    # if predictions csv does not contain scores, add them                                                                                                        "."))
    try:
        bleurt_scores = df_predictions[BLERT_SCORES]
    except KeyError:
        bleurt_scores, bleurt_mean_score, bleurt_best_df, bleurt_worst_df = evaluate(metric_key="bleurt",
                                                                                     generated=generated_text,
                                                                                     references=reference_text,
                                                                                     questions=df_predictions[
                                                                                         "Questions"])
        # save new csv with scores
        df_predictions["bleurt_scores"] = bleurt_scores
        df_predictions.to_csv(DEV_PREDICTIONS_CSV_PATH.replace(".csv", "_with_BLEURT_scores.csv"))

    # no_hops_in_reference_vs_score(no_hops_reference=no_explanations_reference,
    #                               scores=bleurt_scores)
    # no_explanations_in_reference_vs_no_explanations_in_generated(no_explanations_reference=no_explanations_reference,
    #                                                              no_explanations_generated=no_explanations_generated)

    # no_words_in_question_vs_score(questions=df_predictions["Questions"],
    #                               scores=bleurt_scores)

    show_plots()
