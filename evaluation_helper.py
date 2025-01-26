import os
import re
import sys

import pandas as pd

output_file = "output/" + re.sub(r'.+/', '', sys.argv[0][:-3])


def get_datasets():
    for input_file in os.listdir("data/"):
        if input_file.endswith("csv"):
            data = pd.read_csv(f"data/{input_file}", sep="\t")
            yield input_file, data


def save_results(input_file, data):
    data[["y_pred"]].to_csv(f"{output_file}_model_on_{input_file}.csv", index=False, sep="\t")
