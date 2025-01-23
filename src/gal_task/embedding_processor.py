import pandas as pd
import ray
from loguru import logger
from more_itertools import chunked

from gal_task.settings import settings


@ray.remote
class EmbeddingProcessor:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def lookup_word_embedding(self, word, is_complex=False):
        if not is_complex:
            return self.embedding_model.get_vector(word) if word in self.embedding_model else None
        else:
            raise NotImplementedError("Complex words not supported")

    def lookup_phrase_embedding(self, phrase, is_complex=False):
        phrase_embeddings = [self.lookup_word_embedding(word, is_complex) for word in phrase.split(" ")]
        phrase_embeddings = [x for x in phrase_embeddings if x is not None]
        phrase_embedding_normalized = sum(phrase_embeddings) / len(phrase_embeddings)
        return phrase_embedding_normalized

    def lookup_phrases_embeddings(self, phrases, is_complex=False):
        return [self.lookup_phrase_embedding(phrase, is_complex) for phrase in phrases]


class EmbeddingProcessorManager:
    def __init__(self, embedding_model):
        logger.info("Creating Embedder")
        embedding_model_handle = embedding_model.ray_embedding_model_handle
        logger.info(f"Embedding model handle: {embedding_model_handle}")

        self.actors = [EmbeddingProcessor.remote(embedding_model_handle) for _ in range(settings.ray_actors)]

    def get_embeddings(self, phrases_df: pd.DataFrame):
        input_batches = list(chunked(phrases_df["phrases"].tolist(), settings.ray_actors))

        output_batches = [
            actor.lookup_phrases_embeddings.remote(batch) for actor, batch in zip(self.actors, input_batches)
        ]

        results_batched = [ray.get(result) for result in output_batches]

        results_flattened = [embedding for result_batch in results_batched for embedding in result_batch]

        phrases_df["embedding"] = results_flattened
        phrases_df.columns = ["phrases", "embedding"]

        return phrases_df
