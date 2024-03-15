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

import harmony
import pandas as pd

all_questions_seen = []
all_idxs = []
for instrument_name, instrument in harmony.example_instruments.items():
    if "GAD" in instrument_name:
        for question_idx, question in enumerate(instrument.questions):
            all_questions_seen.append(question.question_text)
            all_idxs.append(question_idx)

text_1 = []
text_2 = []
matches = []
for q1_idx in range(len(all_questions_seen)):
    for q2_idx in range(q1_idx + 1, len(all_questions_seen)):
        text_1.append(all_questions_seen[q1_idx])
        text_2.append(all_questions_seen[q2_idx])

        cat1 = all_idxs[q1_idx]
        cat2 = all_idxs[q2_idx]

        is_match = int(cat1 == cat2)

        matches.append(is_match)

df = pd.DataFrame({"text_1": text_1, "text_2": text_2, "y": matches})

df.to_csv(f"gad_7_multilingual.csv", index=False, sep="\t")
