from collections import defaultdict
from collections.abc import Callable
from math import tanh
from matplotlib import ticker, pyplot, colors

import numpy as np
import pandas as pd
import spacy
from dateutil import parser
from numpy import dot
from numpy.linalg import norm
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
DATA_PATH = "data.tsv"
ONE_MONTH_IN_SECONDS = 2764800

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sp = spacy.pipeline.EntityLinker


def load_data(path: str):
    return pd.read_csv(path, sep="|", header=0)


def cosine_sim(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def jaccard_sim(s1, s2):
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def clean_string(string: str) -> str:
    """
    Removes unicode characters, unnecessary punctuation, etc.
    Didn't implement for now as it's very task dependent.
    """
    return string


def split_string_into_sentences(string: str) -> list[str]:
    splitter = SentenceSplitter("en_GB")
    sentences = splitter.split(string)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        yield sentence


def get_entities(string: str) -> list:
    doc = nlp(string)
    return set([str(ent) for ent in doc.ents])


def generate_embeddings_from_string(string):
    string = clean_string(string)
    embeddings = model.encode(string)
    return embeddings


def calculate_pairwise_similarities_from_series(
    series, metric: Callable = cosine_sim
) -> dict[tuple[str, str], int]:
    similarities: dict[tuple[str, str], int] = {}
    for index_1, entity_tensor_1 in enumerate(series):
        for index_2, entity_tensor_2 in enumerate(series):
            if index_1 == index_2:
                similarities[(index_1, index_2)] = 1
            if (index_1, index_2) in similarities:
                continue
            similarity = metric(entity_tensor_1, entity_tensor_2)
            similarities[(index_1, index_2)] = similarity
    return similarities


def calculate_ensemble_probability(
    cosine_similarities: dict[tuple[str, str], int],
    jaccard_similarities: dict[tuple[str, str], int],
    publishing_date_similarities: dict[tuple[str, str], int],
    cosine_weight: float,
    jaccard_weight: float,
    publishing_date_weight: float,
) -> dict[tuple[str, str], int]:
    # TODO: Could be made abstracted/refactored a lot more, but this shows the idea quite nicely
    # also would be nice to have a way to verify if weights sum up to 1

    ensemble_probabilities = defaultdict(float)

    for index_tuple, score in cosine_similarities.items():
        ensemble_probabilities[index_tuple] += score * cosine_weight

    for index_tuple, score in jaccard_similarities.items():
        ensemble_probabilities[index_tuple] += score * jaccard_weight

    for index_tuple, score in publishing_date_similarities.items():
        ensemble_probabilities[index_tuple] += score * publishing_date_weight

    return ensemble_probabilities


def get_duplication_probabilities(events: pd.DataFrame):
    # Article level embeddings & Cosine sim
    events["embedding"] = events.apply(lambda x: generate_embeddings_from_string(x.text), axis=1)
    event_cosine_similarities = calculate_pairwise_similarities_from_series(events["embedding"])

    # NER & Jaccard
    events["entities"] = events.apply(lambda x: get_entities(x.text), axis=1)
    event_jaccard_similarities = calculate_pairwise_similarities_from_series(
        events["entities"], metric=jaccard_sim
    )

    publishing_date_similarities = calculate_pairwise_similarities_from_series(
        events["publishing_date"], metric=get_date_similarity
    )

    ensemble_probabilities = calculate_ensemble_probability(
        event_cosine_similarities,
        event_jaccard_similarities,
        publishing_date_similarities,
        0.4,
        0.4,
        0.2,
    )

    print(ensemble_probabilities)
    return ensemble_probabilities


def get_date_similarity(date_1: str, date_2: str, similarity_scale=ONE_MONTH_IN_SECONDS):
    # If one month difference then similarity = tanh(1) = ~0.24
    parsed_date_1 = parser.parse(date_1)
    parsed_date_2 = parser.parse(date_2)

    difference = abs(parsed_date_1 - parsed_date_2).total_seconds()
    similarity = 1 - abs(tanh((1 / similarity_scale) * difference))

    return similarity


def populate_matrix(probabilities: dict[tuple[int, int], float]) -> np.matrix:
    length = max([prob[0] for prob in probabilities.keys()]) + 1
    matrix = np.zeros((length, length))

    for indices in probabilities.keys():
        matrix[indices[0]][indices[1]] = probabilities[(indices[0], indices[1])]

    return matrix


def save_matrix_as_heatmap(probabilities: np.matrix, filename: str) -> None:
    _, ax = pyplot.subplots()

    heatmap = ax.imshow(probabilities, cmap='inferno', interpolation="nearest")
    cbar = ax.figure.colorbar(heatmap, ax=ax)

    ax.set_title("Ensemble Probabilities")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar.ax.set_ylabel("Probability")

    # Set integer ticks on both axes
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Save the plot to the specified file
    pyplot.savefig(filename)


def main() -> None:
    entities = load_data(DATA_PATH)
    duplication_probabilities = get_duplication_probabilities(entities)
    matrix = populate_matrix(duplication_probabilities)
    save_matrix_as_heatmap(matrix, "similarities.png")


if __name__ == "__main__":
    main()
