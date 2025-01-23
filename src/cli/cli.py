from pathlib import Path

import numpy as np
import pandas
import typer

import gal_task.main
from gal_task.embedding_model import EmbeddingModelSimple
from gal_task.io import get_input_phrases

app = typer.Typer()

embedding_model = EmbeddingModelSimple()


@app.command()
def hello_world() -> None:
    print("hello world")


@app.command()
def embed_phrases_list_to_disk(output_phrase_path) -> None:
    input_phrases = get_input_phrases()
    embedding_model.embed_phrases(input_phrases)
    embedding_model.save_phrases(output_phrase_path)


# @app.command()
# def embed_phrases_list_to_disk(output_phrase_path) -> None:
#     input_phrases = get_input_phrases()
#     embedding_model.embed_phrases(input_phrases)
#     embedding_model.save_phrases(output_phrase_path)


@app.command()
def save_similarities_to_disk(output_similarities_path: Path) -> None:
    word_embeddings = {
        phrase: gal_task.main.encode_phrase(embedding_model.embedding_model, phrase) for phrase in get_input_phrases()
    }
    phrase_embeddings = {
        phrase: gal_task.main.normalized_sum(embeddings) for phrase, embeddings in word_embeddings.items()
    }

    phrase_similarities = [
        (phrase_1, phrase_2, embedding_1, embedding_2, np.linalg.norm(embedding_1 - embedding_2, ord=2))
        for phrase_1, embedding_1 in phrase_embeddings.items()
        for phrase_2, embedding_2 in phrase_embeddings.items()
    ]

    pandas.DataFrame(phrase_similarities).to_csv(output_similarities_path)


@app.command()
def get_phrase_similarity():
    pass


if __name__ == "__main__":
    app()
