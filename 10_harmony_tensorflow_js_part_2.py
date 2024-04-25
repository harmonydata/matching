import json
import re

import harmony
import numpy as np
from harmony.schemas.requests.text import Instrument, Question

import evaluation_helper

with open("10_tensorflow_js_vectors.json", "r", encoding="utf-8") as f:
    question_to_vector = json.loads(f.read())


def convert_texts_to_vector(texts):
    vectors = []
    for text in texts:
        if text in question_to_vector:
            vec_str = question_to_vector[text]
            vec_str = re.sub(r'.+=\s+\\*\s*', '', vec_str)
            vec = json.loads(vec_str)
        else:
            vec = [0.00001] * 512
        vectors.append(np.asarray(vec))
    return np.asarray(vectors)


re_tokeniser = re.compile(r'([a-z]+)')

for input_file, data in evaluation_helper.get_datasets():
    all_questions = list(sorted(set(data.text_1).union(set(data.text_2))))
    question_text_to_idx = dict([b, a] for a, b in enumerate(all_questions))
    questions = []
    for idx, question_text in enumerate(all_questions):
        questions.append(Question(question_text=question_text, question_no=f"{idx}"))
    instrument = Instrument(questions=questions)
    questions, similarity, query_similarity, new_vectors_dict = harmony.match_instruments_with_function([instrument],
                                                                                                        None,
                                                                                                        convert_texts_to_vector)
    preds = [0] * len(data)
    for idx in range(len(data)):
        text_1 = data.text_1.iloc[idx]
        text_2 = data.text_2.iloc[idx]
        idx_1 = question_text_to_idx[text_1]
        idx_2 = question_text_to_idx[text_2]
        preds[idx] = np.abs(similarity[idx_1, idx_2])

    data["y_pred"] = preds

    print(data["y_pred"].isna().sum())

    evaluation_helper.save_results(input_file, data)
