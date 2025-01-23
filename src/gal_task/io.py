import pandas as pd
import pandera as pa

from gal_task.settings import settings


def get_input_phrases():
    with open(settings.default_phrases_path, encoding=settings.default_phrases_encoding) as f:
        phrases = (x.replace("\n", "") for x in f.readlines())
        # Drop the header
        next(phrases)
        phrases = list(phrases)

    return phrases


def get_input_phrases_df() -> pd.DataFrame:
    schema = pa.DataFrameSchema({"Phrases": pa.Column(pa.String, checks=[])})
    df = pd.read_csv(settings.default_phrases_path, encoding=settings.default_phrases_encoding)
    schema.validate(df)
    return df
