import json
import os

from ..patterns.Rule import Rule
from .CompoundRebusGraphParser import CompoundRebusGraphParser

class PhraseRebusGraphParser:
    def __init__(self):
        with open(f"{os.path.dirname(__file__)}/../../saved/ignore_words.json", "r") as file:
            self._ignore_words = json.load(file)

    def parse(self, phrase):
        phrase_words = [word for word in phrase.split() if word not in self._ignore_words]
        phrase = " ".join(phrase_words)
        print(phrase)

        self._divide_text(phrase)

    def _divide_text(self, phrase):
        relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
        phrase_words = phrase.split()
        divided_words = []

        i = 0
        while i < len(phrase_words):
            word = phrase_words[i]
            if word in relational_keywords:
                divided_words.append(phrase_words[:i])
                divided_words.append([phrase_words[i]])
                phrase_words = phrase_words[i+1:]
                i = 0
            i += 1
        divided_words += [phrase_words]

    def _preprocess_text_for_compounds(self, phrase):
        compound_parser = CompoundRebusGraphParser()
        relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]

        phrase_words = phrase.split()
        for i in range(len(phrase_words)-1):
            w1, w2 = phrase_words[i], phrase_words[i+1]
            if w1 in relational_keywords or w2 in relational_keywords:
                continue
            graph = compound_parser.parse(w1, w2, False)[0]
            print(graph)
