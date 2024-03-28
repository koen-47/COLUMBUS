import json

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load pre-trained BERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')


def most_similar_tfidf(input_phrase, phrase_set, num_results=50):
    all_phrases = list(phrase_set)
    all_phrases.append(input_phrase)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_phrases)

    input_index = len(all_phrases) - 1
    similarities = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix[:-1])

    similar_indices = similarities.argsort()[0][::-1][:num_results]
    return [all_phrases[i] for i in similar_indices]


def most_similar_bert(input_string, string_set, num_results=50):
    input_embedding = model.encode(input_string)
    similarities = [(model.encode(s).dot(input_embedding), s) for s in tqdm(string_set, desc="Encoding with BERT")
                    if s != input_string]
    similarities.sort(reverse=True)
    return [s for _, s in similarities[:num_results]]


with open("../saved/idioms_raw.json", "r") as file:
    idioms = json.load(file)

idiom = "all all"
print(idiom)
print(most_similar_bert(idiom, idioms))
print(most_similar_tfidf(idiom, idioms))
