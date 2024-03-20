import json

import pandas as pd

df = pd.read_csv("../saved/ladec_raw.csv")
df_all = df[(df["correctParse"] == "yes")
            & (df["profanity_stim"] == False)
            & (df["profanity_c1"] == False)
            & (df["profanity_c2"] == False)]
df_common = df_all[df_all["isCommonstim"] == 1]

df_all = df_all[["c1", "c2", "stim", "isPlural"]]
df_common = df_common[["c1", "c2", "stim", "isPlural"]]

df_all.to_csv("../saved/ladec_raw_small.csv", index=False)
df_common.to_csv("../saved/ladec_common_raw_small.csv", index=False)

with open("../saved/theidioms_raw.json", "r") as file:
    idioms = pd.DataFrame(json.load(file))
    idioms["idiom"].to_csv("../saved/theidioms_raw_small.csv", index=False)
