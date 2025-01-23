# import pandas as pd
#
# from gal_task.embedding_model import EmbeddingModelSimple
#
#
# def test_distributed_embedder_startup(distributed_ray_embedder):
#     assert distributed_ray_embedder.actors is not None
#
#
# def test_distributed_ray_embedder_same_as_local(embedding_model, distributed_ray_embedder, original_phrases_df):
#     local_embedder = EmbeddingModelSimple()
#     local_embedder.load_all()
#
#     embedded_phrases_df_from_local = local_embedder.embed_phrases(original_phrases_df)
#
#     embedded_phrases_df_from_distributed = distributed_ray_embedder._build_embeddings(original_phrases_df)
#
#     pd.testing.assert_frame_equal(embedded_phrases_df_from_local, embedded_phrases_df_from_distributed)
