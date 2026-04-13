from underthesea import word_tokenize, ner
import re
from symspellpy import SymSpell, Verbosity




class Probability:
    def __init__(self):
        self.sym_spell_2gram = self.sym_spell_3gram = self.sym_spell = SymSpell(
            max_dictionary_edit_distance=2,
            prefix_length=7,
            count_threshold=1
        )
        self.sym_spell.load_dictionary("dictionary/frequency_vi_test.txt", 0, 1, separator="$")
        self.sym_spell_2gram.load_dictionary("dictionary/dic_2_gram_test.txt", 0, 1, separator="$")
        self.sym_spell_3gram .load_dictionary("dictionary/dic_3_gram_test.txt", 0, 1, separator="$")

    def fix_spelling(self, text):
        word_tokenizer_format = word_tokenize(text, format="text")
        word_tokenizer = [token for token in word_tokenizer_format.split()]

        list_suggestions = self.word_tokenizer_suggestions(word_tokenizer)

        list_suggestions.insert(0, [["", -1, -1]])
        list_suggestions.append([["", -1, -1]])
        # Lọc danh sách ứng viên
        for i, suggestion in enumerate(list_suggestions):
            if i < len(list_suggestions) - 1 and i > 0:
                if len(suggestion) > 1:
                    max ,sum_count = self.fix_spelling_word(suggestion,
                        list_suggestions[i - 1][0][0] if list_suggestions[i - 1][0][1] != -1 else list_suggestions[i - 2][0][0],
                        list_suggestions[i + 1])

                    if max[0] == "right":
                        word = max[1]
                        list_suggestions[i] = [[word, 0, 99999]]
                        list_suggestions.remove(list_suggestions[i + 1])
                    else:
                        word = max[1]
                        list_suggestions[i] = [[word, 0, 99999]]
                        list_suggestions.remove(list_suggestions[i - 1])

        result_list = [suggestions_text[0][0].replace("_", " ") for suggestions_text in list_suggestions if len(suggestions_text) !=0]
        result = " ".join(result_list).strip()

        return result

    def word_tokenizer_suggestions(self, word_tokenizer):
        list_suggestions = []
        # Tìm danh sách từ ứng viên
        for i, word in enumerate(word_tokenizer):
            if self.is_valid_token(word):
                if len(word.split()) == 1:
                    suggestions = self.sym_spell.lookup(
                        word.lower(), Verbosity.CLOSEST, max_edit_distance=2
                    )
                else:
                    suggestions = self.sym_spell.lookup_compound(
                        word.lower(), 2
                    )
                suggestion = [[s.term, s.distance, s.count] for s in suggestions]
                if len(suggestion) == 0:
                    word_split = word.replace("_", " ").split()
                    for split in word_split:
                        suggestion_split = self.sym_spell.lookup(
                            split.lower(), Verbosity.CLOSEST, max_edit_distance=2
                        )
                        suggestion = [[s.term, s.distance, s.count] for s in suggestion_split]
                        if len(suggestion) != 0:
                            list_suggestions.append(suggestion)
                        else:
                            list_suggestions.append([[split, 0, 0]])
                else:
                    list_suggestions.append(suggestion)
            else:
                list_suggestions.append([[word, 0, -1]])

        return list_suggestions

    def fix_spelling_word(self, suggestion, left_word, right_suggestions):
        max = ["", "", -1]
        sum_count = 0
        list_probability = []
        for s in suggestion:
            right_corresponds = ["{}_{}".format(s[0].replace(" ", "_"), left_text[0].replace(" ", "_")) for left_text in right_suggestions]
            left_correspond = ["{}_{}".format(left_word.replace(" ", "_"), s[0].replace(" ", "_"))]
            tri_correspond = ["{}_{}".format(left_word.replace(" ", "_"), right_correspond) for right_correspond in right_corresponds]


            left_probability = self.probability_2gram(left_correspond)
            right_probability = self.probability_2gram(right_corresponds)
            tri_probability = self.probability_3gram(tri_correspond)

            list_probability.append([left_probability, "left"])
            list_probability.append([right_probability, "right"])
            list_probability.append([tri_probability, "tri"])

            sum_count += left_probability[2] + right_probability[2] + tri_probability[2]
        for proba in list_probability:
            P = proba[0][1] / sum_count if proba[0][1] > 0 else 0
            if P > max[2]:
                max = [proba[1], proba[0][0], P]
        return max, sum_count

    def probability_2gram(self, words):
        max = -1
        best_word = ""
        sum_count = 0
        for word in words:
            suggestions = self.sym_spell_2gram.lookup(
                word, Verbosity.CLOSEST, max_edit_distance=2
            )
            sum_count += self.count_word(suggestions)
            if self.count_word(suggestions) > max:
                max = self.count_word(suggestions)
                best_word = word
        result = [best_word, max, sum_count]
        return result

    def probability_3gram(self, words):
        max = -1
        best_word = ""
        sum_count = 0
        for word in words:
            suggestions = self.sym_spell_3gram.lookup(
                word, Verbosity.CLOSEST, max_edit_distance=2
            )
            sum_count += self.count_word(suggestions)
            if self.count_word(suggestions) > max:
                max = self.count_word(suggestions)
                best_word = word
        result = [best_word, max, sum_count]
        return result

    def count_word(self, suggestions):
        for suggestion in suggestions:
            if suggestion.distance == 0:
                return suggestion.count
        return 0

    def is_valid_token(self, token):
        if re.match(r'^\d+(\.\d+)?$', token):
            return False

        if "_" in token:
            if ner(token.replace("_"," "))[0][1] == "Np":
                return False

        if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", token):
            return False

        elif re.match(r"^(0[3|5|7|8|9])([0-9]{8})$", token):
            return False

        return all(c.isalpha() or c == '_' for c in token)

if __name__ == "__main__":
    probability = Probability()
    result = probability.fix_spelling("Có kinh nghiệm lý đội nhóm gổm 20 nhan vien cấp dưởi. Tính tổ chửc và kỷ luật cao, không ngửng đổi mởi để cải thiện sản phẩm và dịch vụ đi kèm Luôn đặt trải nghiệm của khách hàng lên trên cùng và nỗ lực đưa tên tuổi thuong hiệu tiếp can đưạc tệp người rộng hon. rong quản dung ")
    print(result)

