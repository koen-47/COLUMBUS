import json

import inflect
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from util import get_node_attributes, get_answer_graph_pairs

inflect = inflect.engine()
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)


def most_similar_jaccard(visible_phrase, phrase_set, split_phrase_set):
    """
    Computes Jaccard similarity for a given input phrase. It measures the similarity between the visible words in a
    puzzles and all phrases/compounds. The compounds are split into their constituent words (i.e., the string "midnight"
    becomes "mid night").

    :param visible_phrase: phrase with the non-visible words in a puzzle removed.
    :param phrase_set: pool of all phrases/compounds.
    :param split_phrase_set: pool of all phrases/compounds, but the compounds are split.
    :return: a dictionary mapping each phrase to a sorted list (in descending) of most similar.
    """
    def _jaccard_sim(phrase_1, phrase_2):
        phrase_1, phrase_2 = set(phrase_1.split()), set(phrase_2.split())
        return len(phrase_1 & phrase_2) / len(phrase_1 | phrase_2)

    similarities = {phrase: _jaccard_sim(split_phrase, visible_phrase) for phrase, split_phrase in
                    zip(phrase_set, split_phrase_set)}
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return dict(similarities)


def get_bert_encodings(phrases):
    """
    Converts all of the specified phrases to a BERT embedding.
    :param phrases: list of phrases to convert to embeddings.
    :return: list of embeddings.
    """
    return [model.encode(phrase) for phrase in tqdm(phrases, desc="Encoding with BERT")]


def most_similar_bert(input_phrase, phrase_set, encodings=None):
    """
    Computes semantic similarity through BERT embeddings for a given input phrase from a pool containing all the
    other phrases/compounds.

    :param input_phrase: phrase/compound to measure most similar other phrases/compounds for.
    :param phrase_set: set of phrases/compounds that will be sampled from.
    :param encodings: BERT encodings for each phrase in phrase_set.
    :return: a dictionary mapping each phrase in phrase_set to its similarity to input_phrase.
    """
    input_embedding = model.encode(input_phrase)
    similarities = np.array([enc.dot(input_embedding) for enc in encodings])
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    phrase_similarities = sorted(dict(zip(phrase_set, similarities)).items(), key=lambda x: x[1], reverse=True)
    return dict(phrase_similarities)


def most_similar_avg(similar_jaccard, similar_bert, lambda_=0.8):
    """
    Computes a weighted average between Jaccard and Sentence-BERT similarity.

    :param similar_jaccard: dictionary mapping all phrases to their similarity to an input phrase through Jaccard
    similarity.
    :param similar_bert: dictionary mapping all phrases to their similarity to an input phrase through Sentence-BERT
    similarity.
    :param lambda_: tuned weight to control the weighted average between Jaccard and Sentence-BERT similarity.
    :return:
    """
    for phrase_jaccard, similarity_jaccard in similar_jaccard.items():
        similarity_bert = similar_bert[phrase_jaccard]
        similarity_avg = lambda_ * similarity_bert + (1 - lambda_) * similarity_jaccard
        similar_bert[phrase_jaccard] = [similarity_bert, similarity_jaccard, similarity_avg]
    most_similar = similar_bert.copy()
    most_similar = sorted(list(most_similar.items()), key=lambda x: x[1][-1], reverse=True)
    return dict(most_similar)


def generate_distractors(phrase_graphs):
    """
    Generates a distractor for each puzzle by extracting the visible text/icons in a puzzle and using Jaccard similarity
    and Sentence-BERT embeddings to sample from a pool of distractors.
    :param phrase_graphs: dictionary mapping each puzzle name to its graph.
    """

    # Load the pool of other idioms/compounds that will be sampled from
    with open("../data/input/idioms_raw.json", "r") as file:
        phrases = json.load(file)
    with open("../data/input/custom_phrases.json", "r") as file:
        phrases += json.load(file)
    compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in pd.read_csv(
        "../data/input/ladec_raw_small.csv").iterrows()}
    custom_compounds = {row["stim"]: f"{row['c1']} {row['c2']}" for _, row in pd.read_csv(
        "../data/input/custom_compounds.csv").iterrows()}
    compounds.update(custom_compounds)
    phrases = phrases + list(compounds.keys())
    split_phrases = phrases + list(compounds.values())

    bert_encodings = get_bert_encodings(phrases)
    answer_to_distractors = {}

    for phrase, graph in list(phrase_graphs.items()):
        # Extract visible words
        visible_words = " ".join([node["text"].lower() if "icon" not in node else list(node["icon"].keys())[0].lower()
                                  for node in list(get_node_attributes(graph).values())])
        phrase_parts = phrase.split("_")
        if phrase_parts[-1].isnumeric():
            phrase_parts = phrase_parts[:-1]
        phrase_ = " ".join(phrase_parts)

        # Compute Jaccard similarity
        similar_jaccard = most_similar_jaccard(visible_words, phrases, split_phrases)

        # Compute Sentence-BERT similarity
        similar_bert = most_similar_bert(phrase_, phrases, encodings=bert_encodings)

        # Compute weighted average between them (lambda = 0.8)
        most_similar = most_similar_avg(similar_jaccard, similar_bert)

        # Get top 10 and top 3 to manually filter through
        most_similar = [p for p in most_similar.keys() if p != phrase_][:10]
        most_similar_top3 = most_similar[:3]
        answer_to_distractors[phrase] = [{"visible": visible_words}, most_similar_top3, most_similar[3:]]

    with open(f"../data/distractors_{model_name.lower()}.json", "w") as file:
        json.dump(answer_to_distractors, file, indent=3)


if __name__ == "__main__":
    puzzles = get_answer_graph_pairs(combine=True)
    generate_distractors(puzzles)
    with open(f"../data/distractors_{model_name.lower()}.json", "r") as file:
        distractors = json.load(file)
        with open(f"../data/distractors_{model_name.lower()}_final.json", "w") as file_2:
            distractors = {phrase: distractor[1] for phrase, distractor in distractors.items()}
            json.dump(distractors, file_2, indent=3)
