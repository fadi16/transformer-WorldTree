import sys

import nltk
import pandas as pd
import numpy as np
from datasets import load_metric
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
###############################################
## todo: change file path
###############################################
from datasets.packaged_modules.csv.csv import Csv

DEV_PREDICTIONS_CSV_PATH = "evaluation/T5-QnA-proper-data-splitting/predictions_vs_actuals-T5-from-QnA-with-data-splitting_with_BLEURT_scores.csv"  # "outputs/dummy_predicions_with_BLEURT_scores.csv"  # "./evaluation/predictions_vs_actuals-T5-from-QnA-with-data-splitting.csv" #"./evaluation/predictions_vs_actuals-T5-from-hypothesis-with-data-splitting.csv"
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


def no_generated_facts_vs_no_facts_in_ref_and_no_repeated_facts(references_with_seperator, generated_with_separator):
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


def jaccard_similarity(s1, s2):
    """
    J(A, B) = |A union B| / (|A| + |B| - |A intersection B|)
    """
    s1 = s1.replace(";", " ").replace(".", " ")
    s2 = s2.replace(";", " ").replace(".", " ")

    s1_bow = set(word.strip() for word in s1.split() if word not in STOP_WORDS and word != "" and not word.isspace())
    s2_bow = set(word.strip() for word in s2.split() if word not in STOP_WORDS and word != "" and not word.isspace())

    s1_bow_inter_s2_bow = s1_bow.intersection(s2_bow)
    score = len(s1_bow_inter_s2_bow) / (len(s1_bow) + len(s2_bow) - len(s1_bow_inter_s2_bow))
    return (1 - score) * 100


def levenshtein_distance(s1, s2):
    return nltk.edit_distance(s1, s2)


def similarity_score_for_QnA_and_reference_vs_BLEURT_score(questions_and_answers, references, scores):
    assert len(questions_and_answers) == len(references) == len(scores)
    similarity_to_score = {}
    for i in range(len(questions_and_answers)):
        similarity = jaccard_similarity(questions_and_answers[i], references[i])
        if similarity in similarity_to_score:
            similarity_to_score[similarity].append(scores[i])
        else:
            similarity_to_score[similarity] = [scores[i]]

    similarities = sorted(list(similarity_to_score.keys()))
    scores = [np.mean(similarity_to_score[s]) for s in similarities]

    figure = get_figure()

    mean_score = np.mean(scores)

    plt.plot(similarities, scores, marker="o", markersize=2, linestyle="solid")
    plt.plot(similarities, [mean_score for _ in similarities])
    plt.title("Similarity Score Between Q/A and Golden Reference vs BLEURT Score of Generated Sentence")
    plt.xlabel("similarity between Q/A and golden reference - levenshtein distance")
    plt.ylabel("bleurt score of generated sentence")
    figure.show()


if __name__ == "__main__":
    df_predictions = pd.read_csv(DEV_PREDICTIONS_CSV_PATH, delimiter=",")

    generated_text = []
    reference_text = []
    no_explanations_reference = []
    no_explanations_generated = []

    reference_text_with_separator = []
    generated_text_with_separator = []

    # with separator
    for x in df_predictions["Generated Text"]:
        generated_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", ""))
    for x in df_predictions["Actual Text"]:
        reference_text_with_separator.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", ""))

    # without separator
    for x in generated_text_with_separator:
        no_explanations_generated.append(x.count("$$") + 1)
        generated_text.append(x.replace("$$", "."))
    for x in reference_text_with_separator:
        no_explanations_reference.append(x.count("$$") + 1)
        reference_text.append(x.replace("$$", "."))

    similarity_score_for_QnA_and_reference_vs_BLEURT_score(questions_and_answers=df_predictions["Questions"],
                                                           references=reference_text,
                                                           scores=df_predictions[BLERT_SCORES])

    show_plots()
    sys.exit()

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

    no_hops_in_reference_vs_score(no_hops_reference=no_explanations_reference,
                                  scores=bleurt_scores)
    no_explanations_in_reference_vs_no_explanations_in_generated(no_explanations_reference=no_explanations_reference,
                                                                 no_explanations_generated=no_explanations_generated)

    no_words_in_question_vs_score(questions=df_predictions["Questions"],
                                  scores=bleurt_scores)

    show_plots()
