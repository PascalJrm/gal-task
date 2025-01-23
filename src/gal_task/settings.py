from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    # TODO: Fix this to pydantic-settings - remove local path
    data_folder: Path = Field(default=Path("/home/p/projects/gal-task/data"))
    input_data_folder: Path = Field(default=Path("/home/p/projects/gal-task/input_data"))

    gensim_model_uri: str = Field(
        default="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=sharing&resourcekey=0-wjGZdNAUop6WykTtMip30g"
    )
    gensim_model_filename: str = Field(default="GoogleNews-vectors-negative300.bin.gz")
    gensim_flat_model_filename: str = Field(default="gensim_model_vectors_flat.csv")
    gensim_flat_model_size: int = Field(default=10000000)

    default_phrases_filename: str = Field(default="phrases.csv")
    default_phrases_encoding: str = Field(default="Windows-1252")
    default_phrases_embedded_filename: str = Field(default="phrases_embedded.csv")

    @property
    def gensim_model_path(self) -> Path:
        return self.data_folder / self.gensim_model_filename

    @property
    def gensim_flat_model_path(self) -> Path:
        return self.data_folder / (str(self.gensim_flat_model_size) + "_" + self.gensim_flat_model_filename)

    @property
    def default_phrases_path(self) -> Path:
        return self.input_data_folder / self.default_phrases_filename

    @property
    def default_phrases_embedded_path(self) -> Path:
        return self.data_folder / (str(self.gensim_flat_model_size) + "_" + self.default_phrases_embedded_filename)


settings = Settings()
