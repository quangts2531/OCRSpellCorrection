from underthesea import word_tokenize, sent_tokenize, text_normalize, ner
import re
import json
from pprintpp import pprint

output_file = "frequency_vi_test.txt"
json_file = "../craw/job_details.json"


def is_valid_token( token):
    if re.match(r'^\d+(\.\d+)?$', token):
        return False


    if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", token):
        return False

    elif re.match(r"^(0[3|5|7|8|9])([0-9]{8})$", token):
        return False

    return all(c.isalpha() or c == '_' for c in token)

seen = set()
with open(json_file, "r", encoding="utf-8") as f:
    jobs = json.load(f)
    words_file = [detail[text] for job in jobs for detail in job["job_detail"] for text in detail]


with open(output_file, "w", encoding="utf-8") as f:
    for i, words in enumerate(words_file):
        print(f"{(i+1)/len(words_file)*100:.2f}%: {i+1}/{len(words_file)}")
        words_sent_tokenize = sent_tokenize(words.replace("\n", ". "))
        for words_sent in words_sent_tokenize:
            words_sent = text_normalize(words_sent)
            save_word = word_tokenize(words_sent, format="text")
            tokens = [token for token in word_tokenize(save_word) if is_valid_token(token)]
            for word in tokens:
                word = word.lower()
                if word not in seen:
                    save_word = word
                    if len(word.replace("_", " ").split()) !=1:
                        for split in word.replace("_", " ").split():
                            if split not in seen:
                                count = sum(text.lower().count(split.lower()) for text in words_file)
                                if count >1000:
                                    f.write(f"{split}${count}\n")
                                    seen.add(split)

                    count = sum(text.lower().count(save_word.replace("_", " ").lower()) for text in words_file)
                    if count > 5:
                        f.write(f"{save_word}${count}\n")
                        seen.add(word)
print("Done:", output_file)