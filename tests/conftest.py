from pathlib import Path

import pytest

from gal_task.distributed_embedding_processor import EmbeddingProcessorManager
from gal_task.embedding_model import EmbeddingModelSimple
from gal_task.io import get_input_phrases_df


@pytest.fixture(scope="session")
def embedding_model():
    embedding_model = EmbeddingModelSimple()
    embedding_model.load_all()
    return embedding_model


@pytest.fixture(scope="session")
def embedding_dict(embedding_model):
    return {
        word: embedding_model.embedding_model_inner[word] for word in embedding_model.embedding_model_inner.index_to_key
    }


@pytest.fixture(scope="session")
def distributed_ray_embedder(embedding_model):
    return EmbeddingProcessorManager(embedding_model)


@pytest.fixture(scope="session")
def original_phrases_df():
    return get_input_phrases_df(Path(__file__).parent / "data" / "original_phrases.csv")


@pytest.fixture(scope="session", autouse=True)
def session_fixture():
    pass
