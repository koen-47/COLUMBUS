import glob
import json
import os
from itertools import islice

import inflect

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from graphs.parsers.RebusGraphParser import RebusGraphParser
from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from util import get_node_attributes, get_answer_graph_pairs

inflect = inflect.engine()
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)


def most_similar_tfidf(input_phrase, phrase_set, num_results=50):
    all_phrases = list(phrase_set)
    all_phrases.append(input_phrase)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_phrases)

    input_index = len(all_phrases) - 1
    similarities = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix[:-1])[0]
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    phrase_similarities = sorted(dict(zip(phrase_set, similarities)).items(),
                                 key=lambda x: x[1], reverse=True)[:num_results]
    return dict(phrase_similarities)


def most_similar_jaccard(input_phrase, visible_phrase, phrase_set, split_phrase_set):
    def _jaccard_sim(phrase_1, phrase_2):
        phrase_1, phrase_2 = set(phrase_1.split()), set(phrase_2.split())
        return len(phrase_1 & phrase_2) / len(phrase_1 | phrase_2)

    similarities = {phrase: _jaccard_sim(split_phrase, visible_phrase) for phrase, split_phrase in
                    zip(phrase_set, split_phrase_set)}
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return dict(similarities)


def get_bert_encodings(phrases):
    return [model.encode(phrase) for phrase in tqdm(phrases, desc="Encoding with BERT")]


def most_similar_bert(input_phrase, phrase_set, encodings=None):
    input_embedding = model.encode(input_phrase)
    similarities = np.array([enc.dot(input_embedding) for enc in encodings])
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    phrase_similarities = sorted(dict(zip(phrase_set, similarities)).items(), key=lambda x: x[1], reverse=True)
    return dict(phrase_similarities)


def most_similar_avg(similar_tfidf, similar_bert, alpha=0.8, num_results=10):
    for phrase_tfidf, similarity_tfidf in similar_tfidf.items():
        similarity_bert = similar_bert[phrase_tfidf]
        similarity_avg = alpha * similarity_bert + (1 - alpha) * similarity_tfidf
        similar_bert[phrase_tfidf] = [similarity_bert, similarity_tfidf, similarity_avg]
    most_similar = similar_bert.copy()
    most_similar = sorted(list(most_similar.items()), key=lambda x: x[1][-1], reverse=True)
    return dict(most_similar)


def generate_distractors(phrase_graphs):
    with open("../saved/idioms_raw.json", "r") as file:
        phrases = json.load(file)
    with open("../saved/custom_phrases.json", "r") as file:
        phrases += json.load(file)
    compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in pd.read_csv("../saved/ladec_raw_small.csv").iterrows()}
    custom_compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in pd.read_csv("../saved/custom_compounds.csv").iterrows()}
    compounds.update(custom_compounds)
    phrases = phrases + list(compounds.keys())
    split_phrases = phrases + list(compounds.values())

    bert_encodings = get_bert_encodings(phrases)
    answer_to_distractors = {}

    for phrase, graph in list(phrase_graphs.items()):
        visible_words = " ".join([node["text"].lower() if "icon" not in node else list(node["icon"].keys())[0].lower()
                                  for node in list(get_node_attributes(graph).values())])
        phrase_parts = phrase.split("_")
        if phrase_parts[-1].isnumeric():
            phrase_parts = phrase_parts[:-1]
        phrase_ = " ".join(phrase_parts)

        similar_jaccard = most_similar_jaccard(phrase_, visible_words, phrases, split_phrases)
        similar_bert = most_similar_bert(phrase_, phrases, encodings=bert_encodings)
        most_similar = most_similar_avg(similar_jaccard, similar_bert)
        most_similar = [p for p in most_similar.keys() if p != phrase_][:10]
        most_similar_top3 = most_similar[:3]
        answer_to_distractors[phrase] = [{"visible": visible_words}, most_similar_top3, most_similar[3:]]

    with open(f"../saved/distractors_{model_name.lower()}.json", "w") as file:
        json.dump(answer_to_distractors, file, indent=3)


# phrase_graphs, compound_graphs = get_answer_graph_pairs()
# phrase_graphs.update(compound_graphs)
# generate_idiom_distractors(phrase_graphs)
# with open(f"../saved/distractors_{model_name.lower()}.json", "r") as file:
#     distractors = json.load(file)
#     with open(f"../saved/distractors_{model_name.lower()}_final.json", "w") as file_2:
#         distractors = {phrase: distractor[1] for phrase, distractor in distractors.items()}
#         json.dump(distractors, file_2, indent=3)
