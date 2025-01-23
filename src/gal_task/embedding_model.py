from typing import Any

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from loguru import logger

from gal_task.io import get_input_phrases
from gal_task.settings import settings


class EmbeddingModelSimple:
    _embedding_model: Any = None
    _phrase_mapping: Any = None

    @property
    def embedding_model(self):
        if not self._embedding_model:
            logger.info("Loading embedding model")
            self._embedding_model = KeyedVectors.load_word2vec_format(
                settings.gensim_model_path, binary=True, limit=settings.gensim_flat_model_size
            )
        return self._embedding_model

    def calculate_static_phrase_mappings(self):
        logger.info("Calculating Static phrase mappings")

        input_phrases = get_input_phrases()

        word_embeddings = self.embed_phrases(input_phrases)

        return word_embeddings

    def save_static_phrase_mappings_to_disk(self):
        logger.info("Saving embedded phrases to disk")

        word_embeddings = self.calculate_static_phrase_mappings()

        df = pd.DataFrame(word_embeddings.items())
        df.columns = ["phrase", "embedding"]

        df.to_parquet(settings.default_phrases_embedded_path)

    def load_static_phrase_mapping_from_disk(self):
        logger.info("Loading embedded phrases from disk")
        return pd.read_parquet(settings.default_phrases_embedded_path)

    @property
    def static_phrase_mappings(self):
        if not settings.default_phrases_embedded_path.exists():
            self.save_static_phrase_mappings_to_disk()

        if self._phrase_mapping is None:
            self._phrase_mapping = self.load_static_phrase_mapping_from_disk()

        return self._phrase_mapping

    def embed_phrase(self, phrase: str):
        def normalized_sum(embeddings: np.ndarray):
            return sum(embeddings) / len(embeddings)

        words = phrase.split()
        # logger.debug(f"The input phrase has length {len(words)}")
        embeddings = np.array([self.embedding_model.get_vector(word) for word in words if word in self.embedding_model])
        return normalized_sum(embeddings)

    def embed_phrases(self, phrases: list[str]) -> dict[str, np.array]:
        logger.info(f"Embedding {len(phrases)} phrases for EmbeddingModel")
        return {phrase: self.embed_phrase(phrase) for phrase in phrases}
