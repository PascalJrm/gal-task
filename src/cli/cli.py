from pprint import pprint

# import pandas as pd
import modin.pandas as pd
import typer
from loguru import logger

from gal_task.embedding_model import EmbeddingModelSimple
from gal_task.settings import settings
from gal_task.similarity import calculate_similarity_between_dataframes, get_most_similar

app = typer.Typer()

embedding_model = EmbeddingModelSimple()


@app.command()
def app_model_status() -> None:
    """Prints the current status of the model"""
    pprint(embedding_model.status)


@app.command()
def check_model() -> None:
    """Checks the model for any known issues - like missing embedding core model"""
    embedding_model.check()
    logger.info("No known issues with embedder")


@app.command()
def calculate_and_save_static_embeddings_to_disk() -> None:
    """Included for completeness with the requested work in process data"""
    embedding_model.save_static_phrase_mappings_to_disk()


@app.command()
def save_cross_similarities_to_disk_as_csv(output_similarities_filename: str) -> None:
    """
    Calculates similarities between the static phrase dictionary and itself. Saves at the output_similarities_path
    :param output_similarities_path: The path to save the similarities csv to
    :return:
    """

    settings.output_data_folder.mkdir(parents=True, exist_ok=True)

    output_similarities_path = settings.output_data_folder / output_similarities_filename

    if not output_similarities_path.parent.exists():
        raise FileNotFoundError()

    logger.info("Calculating similarities")

    similarity_df = calculate_similarity_between_dataframes(
        embedding_model.static_phrase_mappings, embedding_model.static_phrase_mappings
    )

    print(similarity_df)

    similarity_df = similarity_df[["phrases_x", "phrases_y", "similarities"]]

    similarity_df.to_csv(output_similarities_path, index=False)


@app.command()
def get_phrase_similarity(input_phrase: str) -> str:
    """Gets the phrase similarity between a single input phrase and the static phrase dictionary"""
    if len(input_phrase) == 0:
        raise ValueError("Must have a non-empty input phrase")

    input_embedding_df = pd.DataFrame(
        [(input_phrase, embedding_model.embed_phrase(input_phrase))], columns=["phrases", "embedding"]
    )

    similarity_df = calculate_similarity_between_dataframes(embedding_model.static_phrase_mappings, input_embedding_df)

    phrase_1, phrase_2, similarity = get_most_similar(similarity_df)

    output_string = f"The most similar phrase to '{phrase_1}' " f"is '{phrase_2}' with a similarity of '{similarity}'"

    print(output_string)


if __name__ == "__main__":
    app()
