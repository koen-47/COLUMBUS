import json
import re

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open("../saved/rebuses_co_raw.json", "r") as file:
    df = json.load(file)
    df = pd.DataFrame(df)


answers = []
for answer in df["answer"]:
    try:
        answer = answer.split(": ")[1].rstrip(".")
        answers.append(answer.split(".")[0])
    except IndexError:
        pattern = r"\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)+"
        answers.append(re.findall(pattern, answer)[0])
df["answer"] = answers

# type_tags = [tag for tags in df["type_tags"] for tag in tags if tag != "Black & White"]

type_tags = [["Sound", "Position", "Direction"], ["Style", "Numbers", "Colour"], ["Size", "Image", "Highlighting"]]
type_tag = "Size"
puzzles = df[df["type_tags"].apply(lambda x: type_tag in x)]["answer"]
wc = WordCloud(background_color="white").generate(" ".join(puzzles))
plt.imshow(wc)
plt.axis("off")
plt.title(f"Rule: {type_tag}", fontsize=24)

plt.tight_layout()
plt.savefig(f"./rebus_type_wordcloud_{type_tag.lower()}.png")
plt.show()
