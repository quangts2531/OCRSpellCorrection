import random
import os

from sympy.codegen.ast import continue_
from underthesea import word_tokenize, sent_tokenize, text_normalize, ner
import re
import json

output_file = "dic_2_gram_test.txt"
json_file = "../craw/job_details.json"


def is_valid_token(token):
    if re.match(r'^\d+(\.\d+)?$', token):
        return False

    if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", token):
        return False

    elif re.match(r"^(0[3|5|7|8|9])([0-9]{8})$", token):
        return False

    return all(c.isalpha() or c == '_' for c in token)

seen = set()
with open(json_file, "r", encoding="utf-8") as words_file:
    jobs = json.load(words_file)
    words_file = [detail[text] for job in jobs for detail in job["job_detail"] for text in detail]

word_file = []
for i, words in enumerate(words_file):
    print(f"Load {(i+1)/len(words_file)*100:.2f}%: {i+1}/{len(words_file)}")
    words_sent_tokenize = sent_tokenize(words.replace("\n", ". "))
    for words_sent in words_sent_tokenize:
        words_sent = text_normalize(words_sent)
        save_word = word_tokenize(words_sent)
        tokens = [token.lower().replace(" ", "_") for token in save_word if is_valid_token(token)]
        if len(tokens) > 1:
            for i in range(len(tokens)-1):
                word_full = tokens[i] + "_" + tokens[i+1]
                if word_full in seen:
                    for line in word_file:
                        if word_full == line["word"]:
                            line["count"] += 1
                            break
                else:
                    word_file.append({
                        "word": word_full,
                        "count": 1,
                    })
                    seen.add(word_full)
        token_splits = [split.lower() for token in tokens for split in token.replace("_", " ").split()]
        if len(token_splits) > 1:
            for i in range(len(token_splits) - 1):
                word_split = token_splits[i] + "_" + token_splits[i + 1]
                if word_split in seen:
                    for line in word_file:
                        if word_split == line["word"]:
                            line["count"] += 1
                            break
                else:
                    word_file.append({
                        "word": word_split,
                        "count": 1,
                    })
                    seen.add(word_split)

if os.path.dirname(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for i, word in enumerate(word_file):
        print(f"Save: {i/len(word_file)*100:.2f}%: {i}/{len(word_file)}")
        if word['count']>5:
            f.write(f"{word['word']}${word['count']}\n")
print("Done:", output_file)