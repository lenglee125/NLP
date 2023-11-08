import json
import os.path

START_OF_SEQUENCE = "<sos>"
END_OF_SEQUENCE = "<eos>"


class WordMap(object):
    def __init__(self):
        self.word_to_digit = dict()
        self.digit_to_word = list()

        self.add_word(START_OF_SEQUENCE)
        self.add_word(END_OF_SEQUENCE)

    def get_eos(self):
        return self.get_digit(END_OF_SEQUENCE)

    def to_on_hot(self, digit: int) -> list:
        return [0.9999 if i == digit else 0.0001 for i in range(len(self.digit_to_word))]

    def add_word(self, word: str) -> int:
        if word not in self.word_to_digit.keys():
            word_id = len(self.digit_to_word)
            self.word_to_digit[word] = word_id
            self.digit_to_word.append(word)
            return word_id
        else:
            return self.word_to_digit[word]

    def get_digit(self, word: str) -> int:
        if word in self.word_to_digit.keys():
            return self.word_to_digit[word]
        else:
            return -1

    def get_word(self, digit: int) -> str:
        if digit < len(self.digit_to_word):
            return self.digit_to_word[digit]
        else:
            return ""

    def save(self, path: str = "./"):
        with open(os.path.join(path, "word_map.json"), "w", encoding="utf-8") as wm:
            json.dump({"word2digit": self.word_to_digit, "digit2word": self.digit_to_word}, wm, indent=4,
                      ensure_ascii=False)

    @staticmethod
    def load(path: str = "./"):
        with open(os.path.join(path, "word_map.json"), "r", encoding="utf-8") as wm:
            map = json.load(wm, )
            this = WordMap()
            this.digit_to_word = map["digit2word"]
            this.word_to_digit = map["word2digit"]

            return this
