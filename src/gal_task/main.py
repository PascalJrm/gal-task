from collections.abc import Generator

import gdown
import gensim
import numpy as np
import pandas as pd
import pandera as pa
from gensim.models import KeyedVectors
from loguru import logger

from gal_task.settings import settings


def startup() -> None:
    logger.info("Initializing application")

    if not settings.data_folder.exists():
        logger.info("Data folder does not exist - creating")
        settings.data_folder.mkdir(parents=True)
        logger.info(f"Data folder created at {settings.data_folder}")
    else:
        logger.info("Data folder found")

    if not settings.gensim_model_path.exists():
        logger.info("Original model not found - downloading")
        # TODO: Fix this downloaders - gdown is not working
        gdown.download(settings.gensim_model_uri, str(settings.gensim_model_path), quiet=False)
        logger.info(f"Original model downloaded at {settings.gensim_model_path}")
    else:
        logger.info(f"Original model found at {settings.gensim_model_path}")

    if not settings.gensim_flat_model_path.exists():
        logger.info("Flat file model not found - creating")
        wv = KeyedVectors.load_word2vec_format(settings.gensim_model_path, binary=True, limit=1000000)
        wv.save_word2vec_format(str(settings.gensim_flat_model_path))
        logger.info(f"Flat file model created at {settings.gensim_flat_model_path}")
    else:
        logger.info(f"Flat file model found at {settings.gensim_flat_model_path}")


# TODO: Do not require original model if CSV exitst
# TODO: Fix all the mypy errors
# TODO: Write intelligent encoding reader
# TODO: Installed pandas - is this a good move?
# TODO: Is ignoring words not in the model the correct approach
# TODO: I assume the model limit is not picking up some standard words


def get_input_phrases_basic_generator() -> Generator[str, None, None]:
    with open(settings.default_phrases_path, encoding=settings.default_phrases_encoding) as f:
        phrases = (x.replace("\n", "") for x in f.readlines())
        # Drop the header
        next(phrases)

    return phrases


def get_input_phrases_df() -> pd.DataFrame:
    schema = pa.DataFrameSchema({"Phrases": pa.Column(pa.String, checks=[])})
    df = pd.read_csv(settings.default_phrases_path, encoding=settings.default_phrases_encoding)
    schema.validate(df)
    return df


def process_embeddings(model, phrases: list[str]):
    for phrase in phrases:
        print(model[phrase])


def get_model():
    return KeyedVectors.load_word2vec_format(settings.gensim_model_path, binary=True, limit=1000000)


def encode_phrase(model: gensim.models.word2vec, phrase: str):
    words = phrase.split()
    logger.debug(f"The input phrase has length {len(words)}")
    embeddings = [model.get_vector(word) for word in words if word in model]
    logger.debug(f"The output embeddings have length {len(embeddings)}")
    logger.debug(f"An embedding has shape {embeddings[0].shape}")
    average_embedding = np.mean(embeddings, axis=0)
    logger.debug(f"The average embedding has shape {average_embedding.shape}")


def run():
    startup()

    # Load input phrases
    phrases = get_input_phrases_basic_generator()
    print(phrases)
