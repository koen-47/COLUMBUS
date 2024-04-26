import os
import glob
import json

import matplotlib.pyplot as plt
from wordfreq import word_frequency

compounds = [os.path.basename(file).split(".")[0] for file in glob.glob("../results/compounds/all/*")]
word_freq = {word: word_frequency(word, "en") for word in compounds}
word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

common_words = [word for word, freq in word_freq.items() if freq > 1e-7]
print(common_words)
print(len(common_words))
