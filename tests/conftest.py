import tempfile
from pathlib import Path

import pytest

from gal_task.settings import Settings


@pytest.fixture()
def settings_with_empty_working_and_output():
    with tempfile.TemporaryDirectory() as temp_working_dir, tempfile.TemporaryDirectory() as temp_output_dir:
        settings = Settings(
            working_data_folder=Path(temp_working_dir),
            output_data_folder=Path(temp_output_dir),
        )

        yield settings


#
# @pytest.fixture(scope="session")
# def embedding_dict(embedding_model):
#     return {
#         word: embedding_model.embedding_model_inner[word] for word in embedding_model.embedding_model_inner.index_to_key
#     }
#
#
# @pytest.fixture(scope="session")
# def distributed_ray_embedder(embedding_model):
#     return EmbeddingProcessorManager(embedding_model)
#
#
# @pytest.fixture(scope="session")
# def original_phrases_df():
#     return get_input_phrases_df(Path(__file__).parent / "data" / "original_phrases.csv")
#
#
# @pytest.fixture(scope="session", autouse=True)
# def session_fixture():
#     pass
