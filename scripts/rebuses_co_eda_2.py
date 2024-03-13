import json
import re

import umap.umap_ as umap
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open("../saved/rebuses_co_raw.json", "r") as file:
    df = json.load(file)
    df = pd.DataFrame(df)


sentences = []
for answer in df["answer"]:
    try:
        answer = answer.split(": ")[1].rstrip(".")
        sentences.append(answer.split(".")[0])
    except IndexError:
        pattern = r"\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)+"
        sentences.append(re.findall(pattern, answer)[0])
labels = [set([tag for tag in tags if tag != "Black & White"]) for tags in df["type_tags"]]

model = SentenceTransformer('bert-base-nli-mean-tokens')
tokenized_sentences = [sentence.split() for sentence in sentences]

word_embeddings = []
for sentence_tokens in tokenized_sentences:
    embeddings = model.encode(sentence_tokens)
    word_embeddings.append(embeddings)

flattened_embeddings = np.vstack(word_embeddings)

label_set = sorted(set.union(*labels))
label_mapping = {label: i for i, label in enumerate(label_set)}

num_labels = len(set.union(*labels))
Y = np.zeros((flattened_embeddings.shape[0], num_labels))
for i, lbls in enumerate(labels):
    for lbl in lbls:
        label_index = label_mapping[lbl]
        Y[i, label_index] = 1

combined_data = np.hstack((flattened_embeddings, Y))

# tsne = TSNE(n_components=2, random_state=42)
# embedding = tsne.fit_transform(combined_data)

embedding = umap.UMAP(random_state=42).fit_transform(combined_data)

plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=np.argmax(Y, axis=1))
plt.title('t-SNE Visualization of Word Clusters with BERT')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Label')
plt.show()
