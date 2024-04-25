import json
import re
import time

from selenium.webdriver.chrome import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import evaluation_helper

driver = webdriver.WebDriver()

driver.get("https://fastdatascience.com/semantic-similarity-with-sentence-embeddings/")

time.sleep(15)
text_element = driver.find_element(By.XPATH, "//textarea[@id='embeddingtext1']")

vec1 = driver.find_element(By.XPATH, "//div[@id='vec1']")

re_tokeniser = re.compile(r'([a-z]+)')

question_to_vector = {}

for input_file, data in evaluation_helper.get_datasets():
    all_questions = list(sorted(set(data.text_1).union(set(data.text_2))))

    for question_idx, question_text in enumerate(all_questions):
        for i in range(100):
            text_element.send_keys(Keys.BACK_SPACE)
        text_element.send_keys(question_text)

        time.sleep(5)

        button = driver.find_element(By.XPATH, "//button[@id='mybutton']")
        button.click()

        time.sleep(5)

        vec1 = driver.find_element(By.XPATH, "//div[@id='vec1']")

        question_to_vector[question_text] = vec1

with open("10a_tensorflow_js_vectors.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(question_to_vector))
