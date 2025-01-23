import polars as pl
from gensim.models import KeyedVectors

from gal_task.settings import Settings


def transform_model_to_embedding_mapping(model: KeyedVectors) -> pl.DataFrame:
    return pl.DataFrame({
        "words": model.key_to_index.keys(),
        "embeddings": [model.get_vector(word) for word in model.key_to_index],
    })


def standardise_column_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({"Phrases": "phrases"})


def split_column(df: pl.DataFrame, input_column_name: str, output_column_name) -> pl.DataFrame:
    return df.with_columns(pl.col(input_column_name).str.split(" ").alias(output_column_name))


def clean_split_column(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    return df


def process_blank_embedding_rows(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    return df.drop_nulls(subset=[column_name])


def expand_embeddings(df: pl.DataFrame, embedding_column_name: str) -> pl.DataFrame:
    for i in range(300):
        df = df.with_columns(pl.col(embedding_column_name).arr.get(i).alias(f"embeddings_{i}"))
    return df


def get_average_embedding(df: pl.DataFrame, group_column_name: str, embedding_column_name: str) -> pl.DataFrame:
    df = expand_embeddings(df, embedding_column_name)

    sums = [pl.col(f"embeddings_{i}").sum().alias(f"sum_embeddings_{i}") for i in range(300)]
    count = pl.col("embeddings").count().alias("count_embeddings")
    df = df.group_by(group_column_name).agg(sums + [count])  # noqa: RUF005

    for i in range(300):
        df = df.with_columns((pl.col(f"sum_embeddings_{i}") / pl.col("count_embeddings")).alias(f"embeddings_{i}"))

    df = df.with_columns(pl.concat_list(["embeddings_" + str(i) for i in range(300)]).alias(embedding_column_name))
    df = df.with_columns(pl.col(embedding_column_name).cast(pl.Array(pl.Float32, 300)).alias(embedding_column_name))
    df = df.select([group_column_name, embedding_column_name])
    return df


def get_similarity_between_dataframes(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    expanded_df1 = expand_embeddings(df1, "embeddings")
    expanded_df2 = expand_embeddings(df2, "embeddings")

    cross_joined_df = expanded_df1.join(expanded_df2, how="cross")

    diff_squared_embeddings = [
        (pl.col(f"embeddings_{i}") - pl.col(f"embeddings_{i}_right")).pow(2).alias(f"embedding_diff_{i}")
        for i in range(300)
    ]
    cross_joined_df = cross_joined_df.with_columns(diff_squared_embeddings)

    columns_to_sum = [f"embedding_diff_{i}" for i in range(300)]

    cross_joined_df = cross_joined_df.with_columns(pl.sum_horizontal(columns_to_sum).sqrt().alias("similarity"))

    cross_joined_df = cross_joined_df.rename({
        "phrases": "phrases_x",
        "phrases_right": "phrases_y",
        "embeddings": "embeddings_x",
        "embeddings_right": "embeddings_y",
    })

    return cross_joined_df.select(["phrases_x", "phrases_y", "embeddings_x", "embeddings_y", "similarity"])


# def build_and_save_embeddings_if_not_exists(settings: Settings, filename: str, embedding_model_name: str) -> pl.DataFrame:
#     output_path = Path(settings.output_data_folder) / (filename + ".embeddings.parquet")
#
#     schema = InputModel.to_schema()
#
#     if output_path.exists():
#         logger.info("Loading embeddings from disk")
#         return pl.read_parquet(filename)
#     else:
#         df_input = get_input_dataframe(filename)
#         df_validated = apply_schema_with_pandera(df_input, schema)
#         embedding_df = get_embedding_mapping(embedding_model_name)
#         embedding_df = build_embedding_dataframe(df_validated, embedding_df)
#         embedding_df.write_parquet(output_path)


def process_embedded_phrases(settings: Settings, input_df: pl.DataFrame, embedding_df: pl.DataFrame) -> pl.DataFrame:
    df_standardised = standardise_column_names(input_df)
    df_split = split_column(df_standardised, "phrases", "words")
    df_exploded = df_split.explode("words")
    df_joined = df_exploded.join(embedding_df, on="words", how="left")
    df_dropped = process_blank_embedding_rows(df_joined, "embeddings")
    df_averaged = get_average_embedding(df_dropped, "phrases", "embeddings")
    return df_averaged


# def get_embeddings(filename: str, model_name: str) -> pl.DataFrame:
#     output_path = Path(settings.output_data_folder) / (filename + ".embeddings.parquet")
#
#     if not output_path.exists():
#         _build_embeddings(filename, model_name)
#
#     return pl.read_parquet(output_path)


# if __name__ == "__main__":
#     df_input = get_input_dataframe("phrases.csv")
#     df_embeddings = get_embedding_mapping("GoogleNews-vectors-negative300.bin.gz")
#     df_embedded = build_embedding_dataframe(df_input, df_embeddings)
#     similarity = get_similarity_between_dataframes(df_embedded, df_embedded)
#     print(similarity)
