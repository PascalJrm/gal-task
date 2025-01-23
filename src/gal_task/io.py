from gal_task.settings import settings


def get_input_phrases():
    with open(settings.default_phrases_path, encoding=settings.default_phrases_encoding) as f:
        phrases = (x.replace("\n", "") for x in f.readlines())
        # Drop the header
        next(phrases)
        phrases = list(phrases)

    return phrases
