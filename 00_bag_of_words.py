import re
import sys

import evaluation_helper

re_tokeniser = re.compile(r'([a-z]+)')


def get_similarity(text_1, text_2):
    toks_1 = set(re_tokeniser.findall(text_1))
    toks_2 = set(re_tokeniser.findall(text_2))
    denominator = len(toks_1.union(toks_2))
    if denominator == 0:
        return 0
    return len(toks_1.intersection(toks_2)) / denominator


output_file = "output/" + re.sub(r'.+/', '', sys.argv[0][:-3])

for input_file, data in evaluation_helper.get_datasets():
    data["y_pred"] = data[["text_1", "text_2"]].apply(lambda x: get_similarity(x.text_1, x.text_2), axis=1)

    evaluation_helper.save_results(input_file, data)
