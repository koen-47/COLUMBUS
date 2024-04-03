import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

model = SentenceTransformer('paraphrase-distilroberta-base-v1')


def most_similar_tfidf(input_phrase, phrase_set, num_results=50):
    all_phrases = list(phrase_set)
    all_phrases.append(input_phrase)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_phrases)

    input_index = len(all_phrases) - 1
    similarities = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix[:-1])[0]
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    phrase_similarities = sorted(dict(zip(phrase_set, similarities)).items(), key=lambda x: x[1], reverse=True)
    return dict(phrase_similarities)


def most_similar_bert(input_phrase, phrase_set, num_results=50):
    input_embedding = model.encode(input_phrase)
    similarities = np.array([model.encode(s).dot(input_embedding) for s in tqdm(phrase_set, desc="Encoding with BERT")])
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    phrase_similarities = sorted(dict(zip(phrase_set, similarities)).items(), key=lambda x: x[1], reverse=True)
    return dict(phrase_similarities)


def most_similar_avg(similar_tfidf, similar_bert, alpha=0.5):
    for phrase_tfidf, similarity_tfidf in similar_tfidf.items():
        similarity_bert = similar_bert[phrase_tfidf]
        similarity_avg = alpha * similarity_bert + (1 - alpha) * similarity_tfidf
        similar_bert[phrase_tfidf] = [similarity_bert, similarity_tfidf, similarity_avg]
    most_similar = similar_bert.copy()
    print(sorted(most_similar.items(), key=lambda x: x[1][2], reverse=True)[:1000])


def exponential_smoothing(series, alpha):
    smoothed = np.zeros_like(series, dtype=float)
    smoothed[0] = series[0]
    for t in range(1, len(series)):
        smoothed[t] = alpha * series[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


with open("../saved/idioms_raw.json", "r") as file:
    idioms = json.load(file)

idiom = "go out on a limb"
visible_idiom = "go limb"

print(idiom)
print(visible_idiom)

word_freq = {}
for idiom in idioms:
    for word in idiom.split():
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1

words, freqs = word_freq.keys(), np.array(list(word_freq.values()))
freqs = np.clip(1 - (freqs - freqs.min()) / (freqs.max() - freqs.min()), 0.2, 0.8)
print(sorted(dict(zip(words, freqs)).items(), key=lambda x: x[1], reverse=True))

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(idioms)
# idf_values = vectorizer.idf_
# word_to_idf = dict(zip(vectorizer.get_feature_names_out(), idf_values))
# words, idf_scores = word_to_idf.keys(), np.array(list(word_to_idf.values()))
# idf_scores = (idf_scores - idf_scores.min()) / (idf_scores.max() - idf_scores.min())
# print(dict(zip(words, idf_scores)))

# similar_bert = most_similar_bert(idiom, idioms)
# similar_tfidf = most_similar_tfidf(visible_idiom, idioms)
# most_similar_avg(similar_tfidf, similar_bert)
