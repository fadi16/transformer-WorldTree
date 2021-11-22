import pandas as pd
import numpy as np
from datasets import load_metric

###############################################
## todo: change file path
###############################################
DEV_PREDICTIONS_CSV_PATH = "./outputs/dummy_predicions.csv"
num_of_best_worst_explanations = 2

###############################################


def evaluate(metric_key: str, questions, references, generated):
    if metric_key == "bleurt":
        metric = load_metric('bleurt', "bleurt-large-512")
    else:
        raise Exception()

    metric.add_batch(predictions=generated, references=references)
    c = metric.compute()
    scores = np.array(c["scores"])
    print(scores)
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
    print(scores_of_best_explanations)
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
    indicies_of_worst_explanations = np.argpartition(scores, num_of_best_worst_explanations)[:num_of_best_worst_explanations]
    questions_with_worst_explanations = questions[indicies_of_worst_explanations]
    actual_explanations_for_worst_explanations = references[indicies_of_worst_explanations]
    worst_explanations = generated[indicies_of_worst_explanations]
    scores_of_worst_explanations = scores[indicies_of_worst_explanations]
    print(scores_of_worst_explanations)

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

    return scores_mean, best_explanations_df, worst_explanations_df

if __name__ == "__main__":
    df_predictions = pd.read_csv(DEV_PREDICTIONS_CSV_PATH, delimiter=",")

    generated_text = []
    reference_text = []

    count = 0
    for x in df_predictions["Generated Text"]:
        generated_text.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace("$$",
                                                                                                              " "))
    count = 0
    for x in df_predictions["Actual Text"]:
        reference_text.append(
            x.replace(",", " ").replace("[", "").replace("]", "").replace("  ", " ").replace("'", "").replace("$$",
                                                                                                              " "))
    bleurt_mean_score, bleurt_best_df, bleurt_worst_df = evaluate(metric_key="bleurt",
                                             generated=generated_text,
                                             references=reference_text,
                                             questions=df_predictions["Questions"])



