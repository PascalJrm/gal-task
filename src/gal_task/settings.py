from pathlib import Path

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_FOLDER = Path(__file__).parent.parent.parent
DOTENV = ROOT_FOLDER / ".env"

logger.debug("location of settings dotenv file: " + str(DOTENV))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV)

    input_data_folder: Path = Field(default=ROOT_FOLDER / "input_data")
    working_data_folder: Path = Field(default=ROOT_FOLDER / "working_data")
    output_data_folder: Path = Field(default=ROOT_FOLDER / "output_data")

    gensim_model_uri: str = Field(
        default="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=sharing&resourcekey=0-wjGZdNAUop6WykTtMip30g"
    )

    gensim_model_filename: str = Field(default="GoogleNews-vectors-negative300.bin.gz")
    gensim_flat_model_size: int = Field(default=1_000_000)

    default_phrases_filename: str = Field(default="original_phrases.csv")
    default_phrases_encoding: str = Field(default="Windows-1252")
    default_phrases_embedded_filename: str = Field(default="phrases_embedded.parquet")

    use_complex_embeddings: bool = Field(default=False)
