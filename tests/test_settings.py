from gal_task.settings import Settings


def test_load_settings():
    settings = Settings()
    assert settings is not None
