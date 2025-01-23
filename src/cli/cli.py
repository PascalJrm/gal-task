import sys
from pathlib import Path

import numpy as np
import pandas
import typer
from loguru import logger

import gal_task.utility as utility
from gal_task.embedding_model import EmbeddingModelSimple
from gal_task.io import get_input_phrases

app = typer.Typer()

embedding_model = EmbeddingModelSimple()

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@app.command()
def embed_phrases_list_to_disk(output_phrase_path) -> None:
    logger.info("Load input phrases")
    input_phrases = get_input_phrases()
    logger.info("Input phrases loaded")

    embedding_model.embed_phrases(input_phrases)
    embedding_model.save_phrases(output_phrase_path)


# @app.command()
# def embed_phrases_list_to_disk(output_phrase_path) -> None:
#     input_phrases = get_input_phrases()
#     embedding_model.embed_phrases(input_phrases)
#     embedding_model.save_phrases(output_phrase_path)


@app.command()
def save_similarities_to_disk(output_similarities_path: Path) -> None:
    logger.info("Calculating similarities")

    word_embeddings = {
        phrase: utility.encode_phrase(embedding_model.embedding_model, phrase) for phrase in get_input_phrases()
    }
    phrase_embeddings = {phrase: utility.normalized_sum(embeddings) for phrase, embeddings in word_embeddings.items()}

    phrase_similarities = [
        (phrase_1, phrase_2, embedding_1, embedding_2, np.linalg.norm(embedding_1 - embedding_2, ord=2))
        for phrase_1, embedding_1 in phrase_embeddings.items()
        for phrase_2, embedding_2 in phrase_embeddings.items()
    ]
    logger.info("Similarities calculated")
    df = pandas.DataFrame(phrase_similarities)

    df.columns = ["phrase_1", "phrase_2", "embedding_1", "embedding_2", "similarity"]

    df.to_csv(output_similarities_path, index=False)


@app.command()
def get_phrase_similarity(input_phrase: str) -> str:
    embedding = utility.encode_phrase(embedding_model.embedding_model, input_phrase)
    normalized_embedding = utility.normalized_sum(embedding)

    phrase_embeddings = embedding_model.phrase_mapping

    # phrase_embeddings = {
    #     phrase: utility.normalized_sum(embeddings) for phrase, embeddings in word_embeddings.items()
    # }

    phrase_similarities = [
        (input_phrase, phrase, embedding, np.linalg.norm(normalized_embedding - embedding, ord=2))
        for phrase, embedding in phrase_embeddings.items()
    ]

    most_similar_phrase = min(phrase_similarities, key=lambda x: x[3])

    output_string = (
        f"The most similar phrase to '{input_phrase}' "
        f"is '{most_similar_phrase[1]}' with a similarity of '{most_similar_phrase[3]}'"
    )

    print(output_string)


if __name__ == "__main__":
    app()
