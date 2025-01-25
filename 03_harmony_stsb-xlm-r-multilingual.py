import os

os.environ["HARMONY_SENTENCE_TRANSFORMER_PATH"] = "sentence-transformers/stsb-xlm-r-multilingual"

import re

import harmony
import numpy as np
from harmony.schemas.requests.text import Instrument, Question

import evaluation_helper

re_tokeniser = re.compile(r'([a-z]+)')

for input_file, data in evaluation_helper.get_datasets():
    all_questions = list(sorted(set(data.text_1).union(set(data.text_2))))
    question_text_to_idx = dict([b, a] for a, b in enumerate(all_questions))
    questions = []
    for idx, question_text in enumerate(all_questions):
        questions.append(Question(question_text=question_text, question_no=f"{idx}"))
    instrument = Instrument(questions=questions)
    match_response = harmony.match_instruments([instrument])
    similarity = match_response.similarity_with_polarity
    preds = [0] * len(data)
    for idx in range(len(data)):
        text_1 = data.text_1.iloc[idx]
        text_2 = data.text_2.iloc[idx]
        idx_1 = question_text_to_idx[text_1]
        idx_2 = question_text_to_idx[text_2]
        preds[idx] = np.abs(similarity[idx_1, idx_2])

    data["y_pred"] = preds

    evaluation_helper.save_results(input_file, data)
