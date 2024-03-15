'''
MIT License

Copyright (c) 2023 Ulster University (https://www.ulster.ac.uk).
Project: Harmony (https://harmonydata.ac.uk)
Maintainer: Thomas Wood (https://fastdatascience.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import re

import pandas as pd

INPUT_FILE = "raw_mcelroy_tool/Final harmonised item tool EM.xlsx"

validation_data = {}

for sheet_name in ("Childhood", "Adulthood"):

    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)

    if sheet_name == "Adulthood":
        labels_in_this_sheet = ["Adulthood"] * len(df)
        df = df[df.columns[4:]]
    else:
        labels_in_this_sheet = list(df["Developmental period"])
        df = df[df.columns[5:]]

    all_questions = []
    category_to_id = {}
    for idx in range(0, len(df)):
        questions_in_survey = []
        for column in df.columns:
            cell_content = df[column].iloc[idx]
            if type(cell_content) is str:
                cell_content = re.sub("tiredness/exhaustion", "tiredness or exhaustion", cell_content)
                cell_content = re.sub(r'" or "', '/', cell_content)
                for text in cell_content.split("/"):
                    text = re.sub(r'[^A-Za-z -,]', '', text.strip()).strip()
                    text = re.sub('\s+', ' ', text)
                    text = re.sub(r'^\w+\)|IF SLEEP DISTURBANCES OR DEPRESSION | \(rate .+\)| \(\*note .+\)', '', text)
                    text = text.strip()
                    text = re.sub(r'^\"|\"$', '', text)
                    if text[0:1] == text[0:1].lower():
                        text = text[0:1].upper() + text[1:]
                    category = column.strip()
                    if category not in category_to_id:
                        category_to_id[category] = len(category_to_id)
                    category_id = category_to_id[category]
                    if len(text) > 2:
                        questions_in_survey.append((text, category_id))
        all_questions.append(questions_in_survey)

    case_normalised_to_surface_forms = {}
    for group in all_questions:
        for question, category in group:
            case_norm = re.sub(r'[^a-z]', '', question.lower())
            if case_norm == "":
                continue
            if case_norm not in case_normalised_to_surface_forms:
                case_normalised_to_surface_forms[case_norm] = []
            case_normalised_to_surface_forms[case_norm].append(question)

    question_to_categories = {}
    all_questions_seen = set()

    for group in all_questions:
        for question, category in group:
            case_norm = re.sub(r'[^a-z]', '', question.lower())
            question = case_normalised_to_surface_forms[case_norm][0]
            if question != "":
                all_questions_seen.add(question)
                if question not in question_to_categories:
                    question_to_categories[question] = set()
                question_to_categories[question].add(category)
    all_questions_seen = list(sorted(all_questions_seen))

    question_text_to_id = dict([(b, a) for a, b in enumerate(all_questions_seen)])

    if "" in all_questions_seen:
        print (1)

    text_1 = []
    text_2 = []
    matches = []
    for q1_idx in range(len(all_questions_seen)):
        for q2_idx in range(q1_idx + 1, len(all_questions_seen)):
            text_1.append(all_questions_seen[q1_idx])
            text_2.append(all_questions_seen[q2_idx])

            cats1 = question_to_categories[all_questions_seen[q1_idx]]
            cats2 = question_to_categories[all_questions_seen[q2_idx]]

            is_match = int(len(cats1.intersection(cats2)))

            matches.append(is_match)

    df = pd.DataFrame({"text_1": text_1, "text_2": text_2, "y": matches})

    df.to_csv(f"mcelroy_{sheet_name.lower()}.csv", index=False, sep="\t")
