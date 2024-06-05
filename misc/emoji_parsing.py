import os
import json

import pandas as pd
import spacy

with open("../saved/unicode_emojis.txt", "r", encoding="utf-8") as file:
    emojis = {}
    for line in file:
        if line.startswith("# subgroup"):
            current_category = line.split(":")[1].strip()
        if not line.startswith("#") and line != "\n":
            parts = line.split("#")[1]
            if parts != "\n" and parts != " ":
                emoji = parts.split()[0].strip().strip("\t")
                label = " ".join(parts.split()[1:]).strip()
                if " " not in label and "-" not in label:
                    emojis[(label.lower(),)] = emoji
                    # print(emoji, label.lower())

print(emojis)
with open("../saved/icons_v2.json", "w") as file:
    emojis = {str(labels): emoji for labels, emoji in emojis.items()}
    json.dump(emojis, file, indent=3)

# nlp = spacy.load("en_core_web_sm")

# ladec = pd.read_csv(f"{os.path.dirname(__file__)}/../saved/ladec_raw_small.csv")
# all_words = list(set(ladec["c1"].tolist() + ladec["c2"].tolist()))

# doc = nlp(" ".join(all_words))
# nouns = [token.text for token in doc if token.pos_ == "NOUN"]

# overlap = set(all_words).intersection(set(emojis.keys()))
# print(overlap)
# print(len(overlap))
