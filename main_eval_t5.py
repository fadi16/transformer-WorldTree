import pandas as pd
import numpy as np
from datasets import load_metric

if __name__ == "__main__":
    path_predictions = "./outputs/predictions.csv"

    df_predictions = pd.read_csv(path_predictions,delimiter = ",")

    generated_text = []
    reference_text = []

    count = 0
    for x in df_predictions["Generated Text"]:
      generated_text.append(x.replace(","," ").replace("[","").replace("]","").replace("  "," ").replace("'",""))
      # count += 1
      # if count > 5:
      #   break

    count = 0
    for x in df_predictions["Actual Text"]:
      reference_text.append(x.replace(","," ").replace("[","").replace("]","").replace("  "," ").replace("'",""))
      # count += 1
      # if count > 5:
      #   break

    metric = load_metric('bleurt', "bleurt-large-512")
    metric.add_batch(predictions=generated_text, references=reference_text)
    score = np.mean(metric.compute()["scores"])

    print(score)