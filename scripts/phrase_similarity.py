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

from graphs.RebusGraphParser import RebusGraphParser
from util import get_node_attributes

inflect = inflect.engine()
model = SentenceTransformer('all-MiniLM-L6-v2')


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


def get_bert_encodings(phrases):
    return [model.encode(phrase) for phrase in tqdm(phrases, desc="Encoding with BERT")]


def most_similar_bert(input_phrase, phrase_set, num_results=50, encodings=None):
    input_embedding = model.encode(input_phrase)
    if encodings is not None:
        similarities = np.array([enc.dot(input_embedding) for enc in encodings])
    else:
        similarities = np.array([model.encode(s).dot(input_embedding) for s in tqdm(phrase_set, desc="Encoding with BERT")])
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    phrase_similarities = sorted(dict(zip(phrase_set, similarities)).items(), key=lambda x: x[1], reverse=True)[:num_results]
    return dict(phrase_similarities)


def most_similar_avg(similar_tfidf, similar_bert, alpha=0.85, num_results=10):
    for phrase_tfidf, similarity_tfidf in similar_tfidf.items():
        similarity_bert = similar_bert[phrase_tfidf]
        similarity_avg = alpha * similarity_bert + (1 - alpha) * similarity_tfidf
        similar_bert[phrase_tfidf] = [similarity_bert, similarity_tfidf, similarity_avg]
    most_similar = similar_bert.copy()
    most_similar = sorted(most_similar.items(), key=lambda x: x[1][2], reverse=True)[:num_results]
    return dict(most_similar)


def generate_compound_distractors():
    compounds = pd.read_csv("../saved/ladec_raw_small.csv")
    compounds_separate = [f"{row['c1']} {row['c2']}" for _, row in compounds.iterrows()]
    rebus_parser = RebusGraphParser("../saved/ladec_raw_small.csv")
    bert_encodings = get_bert_encodings(compounds_separate)

    answer_to_distractors = {}

    for _, row in compounds.iterrows():
        c1, c2, compound = row["c1"], row["c2"], row["stim"]
        try:
            graphs = rebus_parser.parse_compound(compound)
            if graphs is not None:
                for graph in graphs:
                    visible_word = graph.nodes[1]["text"].lower()
                    similar_tfidf = most_similar_tfidf(visible_word, compounds_separate, num_results=10)
                    similar_tfidf = {d: sim*10 for d, sim in similar_tfidf.items()}
                    similar_bert = most_similar_bert(visible_word, compounds_separate, num_results=10, encodings=bert_encodings)
                    most_similar = {key: max(similar_tfidf.get(key, 0), similar_bert.get(key, 0)) for
                                    key in set(similar_tfidf) | set(similar_bert)}
                    most_similar = dict(sorted(most_similar.items(), key=lambda x: x[1], reverse=True))

                    to_remove = []
                    for distractor in most_similar.keys():
                        full_distractor = "".join(distractor.split())
                        if compound == full_distractor:
                            to_remove.append(distractor)
                        if inflect.singular_noun(compound) == full_distractor or inflect.singular_noun(full_distractor) == compound:
                            to_remove.append(distractor)
                        for distractor_2 in most_similar.keys():
                            full_distractor_2 = "".join(distractor_2.split())
                            if full_distractor != full_distractor_2:
                                if inflect.singular_noun(full_distractor) == full_distractor_2:
                                    to_remove.append(distractor)

                    most_similar = {k: v for k, v in most_similar.items() if k not in to_remove}
                    most_similar_top_3 = list(islice(most_similar, 3))
                    answer_to_distractors[compound] = ["".join(distractor.split()) for distractor in most_similar_top_3]

        except AttributeError:
            pass

    with open("../saved/compound_distractors_v2.json", "w") as file:
        json.dump(answer_to_distractors, file, indent=3)


def generate_idiom_distractors():
    generated_idioms = [" ".join(os.path.basename(file).split(".")[0].split("_"))
                        for file in glob.glob("../results/idioms/all/*")]
    with open("../saved/idioms_raw.json", "r") as file:
        idioms = json.load(file)

    rebus_parser = RebusGraphParser("../saved/ladec_raw_small.csv")
    bert_encodings = get_bert_encodings(idioms)
    answer_to_distractors = {}

    for idiom in generated_idioms:
        # try:
        graph = rebus_parser.parse_idiom(idiom)
        visible_words = " ".join([node["text"].lower() for node in list(get_node_attributes(graph).values())])
        similar_bert = most_similar_bert(idiom, idioms, num_results=len(idioms), encodings=bert_encodings)
        similar_tfidf = most_similar_tfidf(visible_words, idioms, num_results=len(idioms))
        most_similar = most_similar_avg(similar_tfidf, similar_bert, num_results=11)
        most_similar = [idiom_ for idiom_ in most_similar.keys() if idiom_ != idiom]
        answer_to_distractors[idiom] = most_similar

    with open("../saved/idiom_distractors_v2.json", "w") as file:
        json.dump(answer_to_distractors, file, indent=3)


generate_compound_distractors()
generate_idiom_distractors()


# with open("../saved/idioms_raw.json", "r") as file:
#     idioms = json.load(file)
#
# idiom = "go out on a limb"
# visible_idiom = "go limb"
#
# print(idiom)
# print(visible_idiom)
#
# similar_bert = most_similar_bert(idiom, idioms)
# similar_tfidf = most_similar_tfidf(visible_idiom, idioms)
# most_similar_avg(similar_tfidf, similar_bert)

# word_freq = {}
# for idiom in idioms:
#     for word in idiom.split():
#         if word not in word_freq:
#             word_freq[word] = 0
#         word_freq[word] += 1
#
# words, freqs = word_freq.keys(), np.array(list(word_freq.values()))
# freqs = np.clip(1 - (freqs - freqs.min()) / (freqs.max() - freqs.min()), 0.2, 0.8)
# print(sorted(dict(zip(words, freqs)).items(), key=lambda x: x[1], reverse=True))

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(idioms)
# idf_values = vectorizer.idf_
# word_to_idf = dict(zip(vectorizer.get_feature_names_out(), idf_values))
# words, idf_scores = word_to_idf.keys(), np.array(list(word_to_idf.values()))
# idf_scores = (idf_scores - idf_scores.min()) / (idf_scores.max() - idf_scores.min())
# print(dict(zip(words, idf_scores)))

