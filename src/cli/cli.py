from pprint import pprint

import gdown
import typer
from loguru import logger

from gal_task.polars_io import get_embedded_phrase_similarities, get_embedded_phrases, get_phrases_from_input
from gal_task.polars_transforms import get_similarity_between_dataframes
from gal_task.settings import Settings

app = typer.Typer()


@app.command()
def init() -> None:
    """
    Downloads the gensim model to the input data folder
    """

    logger.info("Loading Settings")
    settings = Settings()

    logger.info(f"Model Filename: {settings.gensim_model_filename}")

    model_path = settings.input_data_folder / (settings.gensim_model_filename + "a")

    if not model_path.exists():
        logger.info("Model not downloaded yet. Downloading now")
        gdown.download(id=settings.gensim_model_file_id, output=str(model_path), quiet=False)
    else:
        logger.info("Model already downloaded")

    settings.input_data_folder.mkdir(parents=True, exist_ok=True)
    settings.working_data_folder.mkdir(parents=True, exist_ok=True)
    settings.output_data_folder.mkdir(parents=True, exist_ok=True)


@app.command()
def get_cross_similarities_for_phrase(phrase_filename: str, model_filename: str) -> None:
    """
    Calculates similarities between the static phrase dictionary and itself. Saves at the output_similarities_path
    :param output_similarities_path: The path to save the similarities csv to
    :return:
    """
    init()

    logger.info("Loading Settings")
    settings = Settings()

    # df = get_embedded_phrases(settings, phrase_filename, model_filename, load_from_cache=True, save_to_cache=True)

    similarities_df = get_embedded_phrase_similarities(
        settings, phrase_filename, model_filename, load_from_cache=True, save_to_cache=True
    )
    pprint(similarities_df)

    # uv run cli get-cross-similarities-for-phrase phrases.csv GoogleNews-vectors-negative300.bin.gz


@app.command()
def get_phrase_similarity(phrase_filename: str, model_filename: str, input_phrase: str) -> None:
    """Gets the phrase similarity between a single input phrase and the static phrase dictionary"""
    if len(input_phrase) == 0:
        raise ValueError("Must have a non-empty input phrase")

    logger.info("Loading Settings")
    settings = Settings()

    df_in = get_phrases_from_input(settings, [input_phrase], model_filename)

    df_embedded = get_embedded_phrases(
        settings, phrase_filename, model_filename, load_from_cache=True, save_to_cache=False
    )

    similarities_df = get_similarity_between_dataframes(df_in, df_embedded)

    # Rename Similarity to distance
    most_similarity = similarities_df.sort("similarity").head(1)
    logger.success("Most Similar Phrase is:")
    pprint(most_similarity)
    # uv run cli get-phrase-similarity phrases.csv GoogleNews-vectors-negative300.bin.gz "hello world"


if __name__ == "__main__":
    app()
