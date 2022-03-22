import os
import sys
from typing import List

import numpy as np
import pandas as pd
from datasets import load_metric
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
from completeness_relevence_eval_utils import FactAlignerAndRelevanceScorer, AlignedFact
from postprocess import *
bleurt_metric = None


def evaluate_bleurt(postprocessed_references, postprocessed_generated):
    global bleurt_metric

    if not bleurt_metric:
        bleurt_metric = load_metric('bleurt', "bleurt-large-512")

    bleurt_metric.add_batch(predictions=postprocessed_generated, references=postprocessed_references)
    c = bleurt_metric.compute()
    scores = np.array(c["scores"])

    return scores


def evaluate_bleu(postprocessed_references, postprocessed_generated, metric):
    weight_dict = {
        "bleu2": (0.5, 0.5),
        "bleu3": (0.333, 0.333, 0.334),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
        "bleu5": (0.2, 0.2, 0.2, 0.2, 0.2)
    }

    weight = weight_dict[metric]
    bleu_scores = []

    for ref, gen in zip(postprocessed_references, postprocessed_generated):
        score = bleu_score.sentence_bleu([regexp_tokenize(ref)], regexp_tokenize(gen), weight,
                                         bleu_score.SmoothingFunction().method7)
        bleu_scores.append(score)

    return bleu_scores


def evaluate_rouge(postprocessed_references, postprocessed_generated, metric, post_sep=" . "):
    new_references = []
    for ref in postprocessed_references:
        new_references.append(ref.replace(post_sep, os.linesep))

    new_generated = []
    for gen in postprocessed_generated:
        new_generated.append(gen.replace(post_sep, os.linesep))

    precision_scores = []
    recall_scores = []
    f_measure_scores = []

    # the tokenizer of the scorer keeps newlines for rougeLsum but removes them for the other metrics - which is what
    # we want
    scorer = rouge_scorer.RougeScorer([metric])
    for ref, gen in zip(new_references, new_generated):
        score = scorer.score(target=ref, prediction=gen)
        precision_scores.append(score[metric].precision)
        recall_scores.append(score[metric].recall)
        f_measure_scores.append(score[metric].fmeasure)

    return precision_scores, recall_scores, f_measure_scores


def evaluate_relevance_and_completeness(question_ids, postprocessed_generated_explanations, post_sep=" . "):
    aligner = FactAlignerAndRelevanceScorer()

    relevance_scores = []
    binary_completeness_scores = []
    completeness_scores = []
    f_measure_relevance_completeness_scores = []
    i = 0
    for qid, generated_exp in zip(question_ids, postprocessed_generated_explanations):
        generated_facts = generated_exp.split(post_sep)
        aligned_facts: List[AlignedFact] = []
        for f in generated_facts:
            aligned_fact = aligner.align_and_score(f, qid)
            aligned_facts.append(aligned_fact)

        # no 2 facts should align to the same fact
        unique_aligned_facts = []
        for f in aligned_facts:
            if f not in unique_aligned_facts:
                unique_aligned_facts.append(f)
        aligned_facts = unique_aligned_facts

        relevance_score = len([aligned_f for aligned_f in aligned_facts if aligned_f.relevance_score > 0]) / len(
            aligned_facts)
        no_relevant_golden_facts = len(aligner.get_golden_facts(qid))
        completeness_score = 1 if no_relevant_golden_facts == 0 else len(
            [aligned_f for aligned_f in aligned_facts if aligned_f.is_golden]) / no_relevant_golden_facts
        binary_completeness_score = 1 if completeness_score == 1 else 0
        f_measure_relevance_completeness_score = 0 if (relevance_score + completeness_score) == 0 else 2 * relevance_score * completeness_score / (relevance_score + completeness_score)

        relevance_scores.append(relevance_score)
        completeness_scores.append(completeness_score)
        binary_completeness_scores.append(binary_completeness_score)
        f_measure_relevance_completeness_scores.append(f_measure_relevance_completeness_score)

        if i % 20 == 0:
            print(i)

        i += 1
    return relevance_scores, completeness_scores, binary_completeness_scores, f_measure_relevance_completeness_scores


if __name__ == "__main__":
    from postprocess import postprocess_explanation, regexp_tokenize

    question_ids = pd.read_csv("./data/v2-proper-data/dev_data_wed.csv", sep="\t")["question_id"]

    df = pd.read_csv("./evaluation/bart-plain-metric-agnostic/bs_4_no_penalty.csv")
    actuals = df["Actual Text"]
    predictions = df["Generated Text"]

    postprocessed_actuals = [postprocess_explanation(actual_exp) for actual_exp in actuals]
    postprocessed_generated = [postprocess_explanation(gen_exp) for gen_exp in predictions]

    relevance_scores, completeness_scores, binary_completeness_scores, f_measure_relevance_completeness_scores = evaluate_relevance_and_completeness(
        question_ids, postprocessed_generated)

    mean_relevance = np.mean(relevance_scores)
    mean_completeness = np.mean(completeness_scores)
    mean_binary_completeness = np.mean(binary_completeness_scores)
    mean_relevance_completeness_f_measure = np.mean(f_measure_relevance_completeness_scores)

    print(f"mean_relevance =\t{mean_relevance}")
    print(f"mean_completeness =\t{mean_completeness}")
    print(f"mean_binary_completeness =\t{mean_binary_completeness}")
    print(f"mean_relevance_completeness_f_measure =\t{mean_relevance_completeness_f_measure}")

    df["relevance_score"] = relevance_scores
    df["completeness_score"] = completeness_scores
    df["binary_completeness_score"] = binary_completeness_scores
    df["f_measure_relevance_completeness_score"] = f_measure_relevance_completeness_scores

    sys.exit()

    # generated_explanations = pd.read_csv("./evaluation/BART-plain/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")["Generated Text"]
    # generated_explanations = pd.read_csv("./evaluation/BART-retrieve-prompt-20hyp-6facts/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")["Generated Text"]
    # generated_explanations = pd.read_csv("./evaluation/BART-chains/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")["Generated Text"]
    generated_explanations = \
    pd.read_csv("./evaluation/BART-chain-retrieve/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")[
        "Generated Text"]
    actual_explanations = \
    pd.read_csv("./evaluation/BART-chain-retrieve/test_predictions_vs_actuals_no_rep_with_bleurt_scores.csv")[
        "Actual Text"]

    question_ids = pd.read_csv("./data/v2-proper-data/test_data_wed.csv", sep="\t")["question_id"]

    # postprocessed_generated_explanations = []
    # for exp in generated_explanations:
    #     postprocessed_exp = postprocess_explanation(exp)
    #     postprocessed_generated_explanations.append(postprocessed_exp)

    postprocessed_gen = [postprocess_explanation(gen) for gen in generated_explanations]
    postprocessed_actuals = [postprocess_explanation(actual) for actual in actual_explanations]

    s = evaluate_bleu(postprocessed_actuals, postprocessed_gen, "bleu4")
    print(np.mean(s))
    p, r, f1 = evaluate_rouge(postprocessed_actuals, postprocessed_gen, "rouge4")
    print(np.mean(p))
    print(np.mean(r))
    print(np.mean(f1))
    sys.exit()

    relevance_scores, completeness_scores, binary_completeness_scores = evaluate_relevance_and_completeness(
        question_ids, postprocessed_generated_explanations, " . ")
    print(f"mean relevance = {np.mean(relevance_scores)}")
    print(f"mean completeness = {np.mean(completeness_scores)}")
    print(f"mean binary completeness = {np.mean(binary_completeness_scores)}")
