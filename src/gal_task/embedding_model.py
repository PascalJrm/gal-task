import csv
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import KeyedVectors
from loguru import logger
from pydantic import Field

from gal_task.settings import settings


class EmbeddingModelSimple:
    embedding_model: Any = Field(default=None)
    phrase_mapping: dict[str, np.array] = Field(default_factory=dict)

    def __init__(self):
        logger.info("Initializing application")

        if not settings.data_folder.exists():
            logger.info("Data folder does not exist - creating")
            settings.data_folder.mkdir(parents=True)
            logger.info(f"Data folder created at {settings.data_folder}")
        else:
            logger.info("Data folder found")

        if not settings.gensim_model_path.exists():
            logger.info("Original model not found - downloading")

            # TODO: Fix this to download the file
            raise NotImplementedError("Download the file")
            # gdown.download("https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g", str(settings.gensim_model_path), quiet=False)

            logger.info(f"Original model downloaded at {settings.gensim_model_path}")
        else:
            logger.info(f"Original model found at {settings.gensim_model_path}")

        # if not settings.gensim_flat_model_path.exists():
        #     logger.info("Flat file model not found - creating")
        #     wv = KeyedVectors.load_word2vec_format(
        #         settings.gensim_model_path, binary=True, limit=settings.gensim_flat_model_size
        #     )
        #     wv.save_word2vec_format(str(settings.gensim_flat_model_path))
        #     logger.info(f"Flat file model created at {settings.gensim_flat_model_path}")
        # else:
        #     logger.info(f"Flat file model found at {settings.gensim_flat_model_path}")
        #
        # self.embedding_model = KeyedVectors.load_word2vec_format(settings.gensim_flat_model_path)

        self.embedding_model = KeyedVectors.load_word2vec_format(
            settings.gensim_model_path, binary=True, limit=settings.gensim_flat_model_size
        )

    def _embed_phrase(self, phrase: str):
        words = phrase.split()
        embeddings = np.array([self.embedding_model.get_vector(word) for word in words if word in self.embedding_model])
        return embeddings

    def embed_phrases(self, phrases: list[str]) -> dict[str, np.array]:
        logger.info(f"Embedding {len(phrases)} phrases for EmbeddingModel")
        self.phrase_mapping = {phrase: self._embed_phrase(phrase) for phrase in phrases}

    def save_phrases(self, phrase_path: Path):
        logger.info("Saving embedded phrases to disk")

        with open(settings.default_phrases_embedded_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            for key, value in self.phrase_mapping.items():
                writer.writerow([key, value])

    # def encode_phrase(self, words: list[str]):
    #     words = phrase.split()
    #
    #     logger.debug(f"The input phrase has length {len(words)}")
    #     embeddings = np.array([self.embedding_model.get_vector(word) for word in words if word in model])
    #     logger.debug(f"The output embeddings have length {len(embeddings)}")
    #     logger.debug(f"An embedding has shape {embeddings[0].shape}")
    #     logger.debug(f"An embedding has sum {sum(embeddings[0])}")
    #     return embeddings
    #
    # def normalized_sum(embeddings: np.ndarray):
    #     return sum(embeddings) / len(embeddings)
