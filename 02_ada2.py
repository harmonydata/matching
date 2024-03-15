import harmony
import numpy as np
import openai
from harmony.schemas.requests.text import Instrument, Question

import evaluation_helper

for input_file, data in evaluation_helper.get_datasets():
    all_questions = list(sorted(set(data.text_1).union(set(data.text_2))))
    question_text_to_idx = dict([b, a] for a, b in enumerate(all_questions))
    questions = []
    for idx, question_text in enumerate(all_questions):
        questions.append(Question(question_text=question_text, question_no=f"{idx}"))
    instrument = Instrument(questions=questions)

    model_name = "text-embedding-ada-002"


    def convert_texts_to_vector(texts):
        batch_size = 100
        embeddings_as_list = []
        texts = list(texts)
        for j in range(len(texts)):
            if texts[j] == "":
                texts[j] = "empty"
        for batch_start in range(0, len(texts), batch_size):
            batch_end = batch_start + batch_size
            if batch_end > len(texts):
                batch_end = len(texts)
            batch = texts[batch_start:batch_end]
            vectors = openai.Embedding.create(input=batch, model=model_name)['data']
            embeddings_as_list.extend([vectors[i]["embedding"] for i in range(len(vectors))])
        return np.asarray(embeddings_as_list)


    all_questions, similarity, query_similarity, new_vectors_dict = harmony.match_instruments_with_function(
        [instrument], None,
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
