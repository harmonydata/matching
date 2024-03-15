from typing import List

import harmony
import numpy as np
from harmony.schemas.requests.text import Instrument
from harmony.schemas.requests.text import Question
from sentence_transformers import SentenceTransformer

import evaluation_helper

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def convert_texts_to_vector(texts: List) -> np.ndarray:
    embeddings = model.encode(sentences=texts, convert_to_numpy=True)

    return embeddings


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

    evaluation_helper.save_results(input_file, data)
