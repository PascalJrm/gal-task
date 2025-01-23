from pathlib import Path

import pandera.polars as pa
import polars as pl
from gensim.models import KeyedVectors
from loguru import logger

from gal_task.polars_transforms import (
    get_similarity_between_dataframes,
    process_embedded_phrases,
    transform_model_to_embedding_mapping,
)
from gal_task.settings import Settings


class InputModel(pa.DataFrameModel):
    Phrases: str


def get_paths_for_input_and_cache(settings: Settings, filename: str) -> tuple:
    input_path = Path(settings.input_data_folder) / filename
    logger.info("Input Path: " + str(input_path))
    input_path_cached = Path(settings.working_data_folder) / (filename + ".cached.parquet")
    logger.info("Input Path Cached: " + str(input_path_cached))

    return input_path, input_path_cached


def get_and_validate_input_phrases_dataframe(
    settings: Settings, filename: str, load_from_cache: bool = True, save_to_cache: bool = True
) -> pl.DataFrame:
    input_path, input_path_cached = get_paths_for_input_and_cache(settings, filename)

    if not input_path_cached.exists():
        logger.info("No Cached Data Found")
    else:
        if load_from_cache:
            logger.info("Cached Data Found: loading from cache")
            return pl.read_parquet(input_path_cached)
        else:
            logger.info("Cached Data Found: not loading from cache")

    df = pl.read_csv(input_path, has_header=True, encoding=settings.default_phrases_encoding)
    # df_validated = df.pipe(InputModel.to_schema().validate)

    if save_to_cache:
        logger.info("Saving Data To Cache")
        df.write_parquet(str(input_path_cached))

    return df


def get_raw_embedding_model(settings: Settings, embedding_model_name: str) -> KeyedVectors:
    input_path = Path(settings.input_data_folder) / embedding_model_name

    if not input_path.exists():
        raise FileNotFoundError()

    keyed_vectors = KeyedVectors.load_word2vec_format(input_path, binary=True, limit=1_000_000)
    logger.info("Loaded Keyed Vectors")

    return keyed_vectors


def get_input_embedding_dataframe(
    settings: Settings, filename: str, load_from_cache: bool = True, save_to_cache: bool = True
) -> pl.dataframe:
    input_path, input_path_cached = get_paths_for_input_and_cache(settings, filename)

    if not input_path_cached.exists():
        logger.info("No Cached Data Found")
    else:
        if load_from_cache:
            logger.info("Cached Data Found: loading from cache")
            return pl.read_parquet(input_path_cached)
        else:
            logger.info("Cached Data Found: not loading from cache")
    embedding_model = get_raw_embedding_model(settings, filename)
    embedding_df = transform_model_to_embedding_mapping(embedding_model)

    if save_to_cache:
        logger.info("Saving Data To Cache")
        embedding_df.write_parquet(str(input_path_cached))

    return embedding_df


def get_embedded_phrases(
    settings: Settings, filename: str, model_filename: str, load_from_cache: bool = True, save_to_cache: bool = True
) -> pl.DataFrame:
    input_path = Path(settings.input_data_folder) / filename
    logger.info("Input Path: " + str(input_path))
    input_path_cached = Path(settings.working_data_folder) / (filename + ".embedded.cached.parquet")
    logger.info("Input Path Cached: " + str(input_path_cached))

    input_df = get_and_validate_input_phrases_dataframe(settings, filename, load_from_cache, save_to_cache)
    embedding_df = get_input_embedding_dataframe(settings, model_filename, load_from_cache, save_to_cache)

    if not input_path_cached.exists():
        logger.info("No Cached Data Found")
    else:
        if load_from_cache:
            logger.info("Cached Data Found: loading from cache")
            return pl.read_parquet(input_path_cached)
        else:
            logger.info("Cached Data Found: not loading from cache")

    embedded_phrases_df = process_embedded_phrases(settings, input_df, embedding_df)

    logger.debug("Number of columns = " + str(len(embedded_phrases_df.columns)))

    if save_to_cache:
        logger.info("Saving Data To Cache")
        embedded_phrases_df.write_parquet(str(input_path_cached))

    return embedded_phrases_df


def get_embedded_phrase_similarities(
    settings: Settings, filename: str, model_filename: str, load_from_cache: bool = True, save_to_cache: bool = True
) -> pl.DataFrame:
    input_path = Path(settings.input_data_folder) / filename
    logger.info("Input Path: " + str(input_path))
    input_path_cached = Path(settings.working_data_folder) / (filename + ".similarities.cached.parquet")
    logger.info("Input Path Cached: " + str(input_path_cached))

    if not input_path_cached.exists():
        logger.info("No Cached Data Found")
    else:
        if load_from_cache:
            logger.info("Cached Data Found: loading from cache")
            return pl.read_parquet(input_path_cached)
        else:
            logger.info("Cached Data Found: not loading from cache")

    # Calculate
    df = get_embedded_phrases(settings, filename, model_filename, load_from_cache, save_to_cache)
    similarities_df = get_similarity_between_dataframes(df, df)

    if save_to_cache:
        logger.info("Saving Data To Cache")
        similarities_df.write_parquet(str(input_path_cached))

    return similarities_df


def get_phrases_from_input(settings: Settings, input_strings: list[str], model_filename: str) -> pl.DataFrame:
    input_df = pl.DataFrame({"Phrases": input_strings})
    print(input_df)
    embedding_df = get_input_embedding_dataframe(settings, model_filename)
    print(embedding_df)
    return process_embedded_phrases(settings, input_df, embedding_df)
