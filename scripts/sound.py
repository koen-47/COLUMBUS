import json

import pandas as pd
from SoundsLike.SoundsLike import Search
from tqdm import tqdm

df = pd.read_csv("../saved/ladec_common_raw_small.csv")
unique_words = set(df["c1"]).union(set(df["c2"]))

perfect_close_sounding_words = {}
for word in tqdm(unique_words, desc="Collecting perfect/close sounding words"):
    try:
        perfect_sounding = set([w.lower() for w in Search.perfectHomophones(word)]).difference({word})
        close_sounding = set([w.lower() for w in Search.closeHomophones(word)]).difference({word})
        perfect_close_sounding_words[word] = {"perfect": list(perfect_sounding), "close": list(close_sounding)}
    except ValueError:
        perfect_close_sounding_words[word] = {"perfect": [], "close": []}


with open("../saved/perfect_close_sounding_words.json", "w") as file:
    json.dump(perfect_close_sounding_words, file, indent=3)
