import nltk
from nltk.corpus import wordnet, wordnet_ic

nltk.download('wordnet', quiet=True)


def get_most_likely_synonyms(word):
    max_similarity = -1
    best_synonyms = []
    word_synsets = wordnet.synsets(word)
    if not word_synsets:
        return None

    base_synset = word_synsets[0]

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    for synset in word_synsets:
        similarity = base_synset.res_similarity(synset, brown_ic)
        if similarity > max_similarity:
            max_similarity = similarity
            best_synonyms = [lemma.name() for lemma in synset.lemmas()]

    return best_synonyms


word = "limb"
synonyms = get_most_likely_synonyms(word)

if synonyms:
    print(f"Synonyms of '{word}':")
    print(", ".join(synonyms))
else:
    print(f"No synonyms found for '{word}'.")