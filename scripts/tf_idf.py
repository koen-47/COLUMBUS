import os

import glob
import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist

import pandas as pd

nltk.download('brown', quiet=True)

# puzzle_words = pd.read_csv("../saved/ladec_raw_small.csv")["stim"]
puzzle_words = [os.path.basename(puzzle).split('.')[0] for puzzle in glob.glob("../results/compounds/all/*")]
puzzle_word_freq = {}

brown_words = [word.lower() for word in brown.words()]
fdist = FreqDist(brown_words)


for word in puzzle_words:
    puzzle_word_freq[word] = fdist[word]

# print(puzzle_word_freq)
print(dict(sorted(puzzle_word_freq.items(), key=lambda item: item[1], reverse=True)))
