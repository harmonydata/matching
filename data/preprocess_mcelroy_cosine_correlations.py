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

import pandas as pd

codebook = pd.read_excel("raw_mcelroy_cosine_correlation/codebook.xlsx")

supp = pd.read_csv("raw_mcelroy_cosine_correlation/SuppFile2.csv", skiprows=1)

item_no_to_text = dict(codebook.set_index("Item number")["Content"])

supp["text_1"] = supp["from"].map(item_no_to_text)
supp["text_2"] = supp["to"].map(item_no_to_text)

supp.rename(columns={"spearman": "y"}, inplace=True)

supp[["text_1", "text_2", "y", "cosine"]].to_csv("mcelroy_cosine_correlation.csv", index=False, sep="\t")
