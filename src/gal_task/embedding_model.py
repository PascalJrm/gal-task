from typing import Any

import numpy as np

# import pandas as pd
import pandas as pd
import ray
from gensim.models import KeyedVectors
from loguru import logger

from gal_task.io import get_input_phrases_df
from gal_task.settings import settings


class EmbeddingModelSimple:
    _embedding_model: Any = None
    _phrase_mapping: Any = None

    _ray_embedding_model_handle: Any = None

    @property
    def status(self):
        return {
            "embedding_model_exists": settings.gensim_model_path.exists(),
            "embedding_model_loaded": bool(self._embedding_model),
            "phrase_mapping_cache_exists": settings.default_phrases_embedded_path.exists(),
            "phrase_mapping_cache_loaded": bool(self._phrase_mapping),
        }

    def check(self):
        if not self.status["embedding_model_exists"]:
            raise FileNotFoundError()
        else:
            logger.info(f"Embedding model found at {settings.gensim_model_path}")

    def load_all(self):
        self.check()
        self.embedding_model_inner.get()
        self.static_phrase_mappings.get()
        self.ray_embedding_model_handle.get()

    @property
    def ray_embedding_model_handle(self):
        if not self._ray_embedding_model_handle:
            ray.init(ignore_reinit_error=True)
            self._ray_embedding_model_handle = ray.put(self.embedding_model_inner)
            logger.debug(
                f"Ray embedding model handle: {self._ray_embedding_model_handle} of type {type(self._ray_embedding_model_handle)}"
            )
        return self._ray_embedding_model_handle

    @property
    def embedding_model_inner(self):
        if not self._embedding_model:
            logger.info("Loading embedding model")
            self._embedding_model = KeyedVectors.load_word2vec_format(
                settings.gensim_model_path, binary=True, limit=settings.gensim_flat_model_size
            )
        return self._embedding_model

    def calculate_static_phrase_mappings(self) -> pd.DataFrame:
        logger.info("Calculating Static phrase mappings")
        input_phrases = get_input_phrases_df()
        word_embeddings = self.embed_phrases(input_phrases)
        return word_embeddings

    def save_static_phrase_mappings_to_disk(self):
        logger.info("Saving embedded phrases to disk")
        word_embeddings = self.calculate_static_phrase_mappings()
        word_embeddings.to_parquet(settings.default_phrases_embedded_path)

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

    def embed_phrase(self, phrase: str) -> np.ndarray:
        def normalized_sum(embeddings: np.ndarray):
            return sum(embeddings) / len(embeddings)

        words = phrase.split()

        if settings.use_complex_embeddings:
            raise NotImplementedError("Complex embeddings not implemented in local")
        else:
            embeddings = np.array([
                self.embedding_model_inner.get_vector(word) for word in words if word in self.embedding_model_inner
            ])

        return normalized_sum(embeddings)

    def embed_phrases(self, phrases_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Embedding {len(phrases_df)} phrases for EmbeddingModel")
        phrases_df["embedding"] = phrases_df.apply(lambda x: self.embed_phrase(x["Phrases"]), axis=1)
        phrases_df.columns = ["phrases", "embedding"]
        return phrases_df
