# import gensim
# import numpy as np
# from gensim.models import KeyedVectors
# from loguru import logger
#
# from gal_task.settings import settings
#
# # TODO: Do not require original model if CSV exitst
# # TODO: Fix all the mypy errors
# # TODO: Write intelligent encoding reader
# # TODO: Installed pandas - is this a good move?
# # TODO: Is ignoring words not in the model the correct approach
# # TODO: I assume the model limit is not picking up some standard words
# # TODO: Move the startup code into the embedding model initiation
# # TODO: Deal with bad input sentence
#
#
# def process_embeddings(model, phrases: list[str]):
#     for phrase in phrases:
#         print(model[phrase])
#
#
# def get_model_csv():
#     logger.debug(f"Loading model from {settings.gensim_flat_model_path}")
#     return KeyedVectors.load_word2vec_format(settings.gensim_flat_model_path)
#
#
# def encode_phrase(model: gensim.models.word2vec, phrase: str):
#     words = phrase.split()
#     logger.debug(f"The input phrase has length {len(words)}")
#     embeddings = np.array([model.get_vector(word) for word in words if word in model])
#     return embeddings
#
#
# def normalized_sum(embeddings: np.ndarray):
#     return sum(embeddings) / len(embeddings)
#
#
# # def run():
# #     startup()
# #
# #     # model = get_model_csv()
# #
# #     embedding_model = EmbeddingModelSimple()
# #
# #     embedding_model.embed_phrases(list(get_input_phrases_basic_generator()))
# #
# #     embedding_model.save_phrases(settings.data_folder / "original_phrases.csv")
#
# # Load input phrases
# # model = get_model()
# # word_embeddings = {phrase: encode_phrase(model, phrase) for phrase in get_input_phrases_basic_generator()}
# # phrase_embeddings = {phrase: normalized_sum(embeddings) for phrase, embeddings in word_embeddings.items()}
# #
# # phrase_similarities = [
# #     (phrase_1, phrase_2, embedding_1, embedding_2, np.linalg.norm(embedding_1 - embedding_2, ord=2))
# #     for phrase_1, embedding_1 in phrase_embeddings.items()
# #     for phrase_2, embedding_2 in phrase_embeddings.items()
# # ]
# #
# # print(phrase_similarities[1])
# #
# # input_phrase = "The premiums are high"
# #
# # encoded_phrase = encode_phrase(model, input_phrase)
# #
# # normalized_phrase = normalized_sum(encoded_phrase)
# #
# # phrase_similarities = [
# #     (phrase, np.linalg.norm(normalized_phrase - embedding, ord=2))
# #     for phrase, embedding in phrase_embeddings.items()
# # ]
# #
# # print(phrase_similarities)
# #
# # print(min(phrase_similarities, key=lambda x: x[1]))
