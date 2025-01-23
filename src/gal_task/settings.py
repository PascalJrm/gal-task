from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    # TODO: Fix this to pydantic-settings - remove local path
    data_folder: Path = Field(default=Path("/home/p/projects/gal-task/data"))
    input_data_folder: Path = Field(default=Path("/home/p/projects/gal-task/input_data"))

    gensim_model_uri: str = Field(default="http://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM")
    gensim_model_filename: str = Field(default="GoogleNews-vectors-negative300.bin.gz")
    gensim_flat_model_filename: str = Field(default="gensim_model_vectors_flat.csv")

    default_phrases_filename: str = Field(default="phrases.csv")
    default_phrases_encoding: str = Field(default="Windows-1252")

    @property
    def gensim_model_path(self) -> Path:
        return self.data_folder / self.gensim_model_filename

    @property
    def gensim_flat_model_path(self) -> Path:
        return self.data_folder / self.gensim_flat_model_filename

    @property
    def default_phrases_path(self) -> Path:
        return self.input_data_folder / self.default_phrases_filename


settings = Settings()
