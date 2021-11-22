import pandas as pd
import numpy as np
from datasets import load_metric

DEV_PREDICTIONS_CSV_PATH = "./outputs/predictions.csv"


if __name__ == "__main__":
    df_predictions = pd.read_csv(DEV_PREDICTIONS_CSV_PATH,delimiter = ",")

    generated_text = []
    reference_text = []

    count = 0
    for x in df_predictions["Generated Text"]:
      generated_text.append(x.replace(","," ").replace("[","").replace("]","").replace("  "," ").replace("'","").replace("$$", " "))
      # count += 1
      # if count > 5:
      #   break

    count = 0
    for x in df_predictions["Actual Text"]:
      reference_text.append(x.replace(","," ").replace("[","").replace("]","").replace("  "," ").replace("'","").replace("$$", " "))
      # count += 1
      # if count > 5:
      #   break

    # TODO: add more metrics
    # TODO: print 10 best generated/actual with scores
    # TODO: print 10 worst generated/actual with scores
    # TODO: this is evaluation for validation data, add evaluation for test data

    metric = load_metric('bleurt', "bleurt-large-512")
    metric.add_batch(predictions=generated_text, references=reference_text)

    scores = metric.compute()["scores"]
    scores_mean = np.mean(metric.compute()["scores"])

    print(scores_mean)