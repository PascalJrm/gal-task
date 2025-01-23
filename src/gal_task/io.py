from pathlib import Path
from typing import Optional

import pandas as pd
import pandera as pa

from gal_task.settings import settings


def get_input_phrases_df(input_path: Optional[Path] = None) -> pd.DataFrame:
    schema = pa.DataFrameSchema({"Phrases": pa.Column(pa.String, checks=[])})
    read_path = input_path or settings.default_phrases_path
    df = pd.read_csv(read_path, encoding=settings.default_phrases_encoding)
    schema.validate(df)
    return df
