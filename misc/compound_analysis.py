import os
import json

import pandas as pd
import spacy

# nlp = spacy.load("en_core_web_sm")

ladec = pd.read_csv(f"{os.path.dirname(__file__)}/../saved/ladec_raw_small.csv")
all_words = list(set(ladec["c1"].tolist() + ladec["c2"].tolist()))

# doc = nlp(" ".join(all_words))
# nouns = [token.text for token in doc if token.pos_ == "NOUN"]

print(json.dumps(all_words, indent=3))
