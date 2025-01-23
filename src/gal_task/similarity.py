import numpy as np
import pandas as pd


def calculate_similarity_between_dataframes(df1, df2):
    phrases_joined = pd.merge(df1, df2, how="cross")

    phrases_joined["similarities"] = phrases_joined.apply(
        lambda x: np.linalg.norm(x["embedding_x"] - x["embedding_y"], ord=2), axis=1
    )

    return phrases_joined


def get_most_similar(similarity_dataframe):
    min_index = similarity_dataframe["similarities"].idxmin()

    maximum_similarity = similarity_dataframe.iloc[min_index]

    return maximum_similarity["phrases_x"], maximum_similarity["phrases_y"], maximum_similarity["similarities"]
