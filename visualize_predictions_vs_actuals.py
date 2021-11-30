import pandas as pd
from rich.table import Column, Table
from rich import box
import rich

#####################
## todo change if needed
######################
PREDICTIONS_CSV_FILE_PATH = "./outputs/actuals_vs_predictions.csv"
######################

QUESTION = "Questions"
ACTUAL = "Actual Text"
GENERATED = "Generated Text"

def display_df(df):
    table = Table(
        Column(QUESTION, justify="center"),
        Column(ACTUAL, justify="center"),
        Column(GENERATED, justify="center"),

        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )
    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[1], row[2], row[3])

    rich.print(table)


if __name__ == "__main__":
    df = pd.read_csv(PREDICTIONS_CSV_FILE_PATH)
    display_df(df)
