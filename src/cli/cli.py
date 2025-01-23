from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from gal_task.embedding_model import EmbeddingModelSimple

app = typer.Typer()

embedding_model = EmbeddingModelSimple()


@app.command()
def calculate_and_save_static_embeddings_to_disk() -> None:
    embedding_model.save_static_phrase_mappings_to_disk()


@app.command()
def save_similarities_to_disk_as_csv(output_similarities_path: Path) -> None:
    logger.info("Calculating similarities")

    phrases_joined = pd.merge(
        embedding_model.static_phrase_mappings, embedding_model.static_phrase_mappings, how="cross"
    )

    phrases_joined["similarities"] = phrases_joined.apply(
        lambda x: np.linalg.norm(x["embedding_x"] - x["embedding_y"], ord=2), axis=1
    )

    phrases_joined.to_csv(output_similarities_path, index=False)


@app.command()
def get_phrase_similarity(input_phrase: str) -> str:
    input_embedding_df = pd.DataFrame(
        [(input_phrase, embedding_model.embed_phrase(input_phrase))], columns=["phrases", "embedding"]
    )
    print(input_embedding_df)

    phrases_joined = pd.merge(embedding_model.static_phrase_mappings, input_embedding_df, how="cross")

    phrases_joined["similarities"] = phrases_joined.apply(
        lambda x: np.linalg.norm(x["embedding_x"] - x["embedding_y"], ord=2), axis=1
    )

    min_index = phrases_joined["similarities"].idxmin()

    maximum_similarity = phrases_joined.iloc[min_index]

    output_string = (
        f"The most similar phrase to '{maximum_similarity.phrases_x}' "
        f"is '{maximum_similarity.phrases_y}' with a similarity of '{maximum_similarity.similarities}'"
    )

    print(output_string)


if __name__ == "__main__":
    app()
