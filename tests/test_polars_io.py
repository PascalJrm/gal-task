from pathlib import Path

from gal_task.polars_io import (
    get_and_validate_input_phrases_dataframe,
    get_embedded_phrases,
    get_input_embedding_dataframe,
)


def count_files_in_folder(folder: Path):
    return len(list(folder.glob("*")))


def test_load_input_dataframe(settings_with_empty_working_and_output):
    assert count_files_in_folder(settings_with_empty_working_and_output.input_data_folder) == 3
    assert count_files_in_folder(settings_with_empty_working_and_output.working_data_folder) == 0
    assert count_files_in_folder(settings_with_empty_working_and_output.output_data_folder) == 0

    df = get_and_validate_input_phrases_dataframe(
        settings_with_empty_working_and_output, "phrases.csv", load_from_cache=True, save_to_cache=True
    )

    assert df is not None

    assert count_files_in_folder(settings_with_empty_working_and_output.working_data_folder) == 1


def test_load_embedding_dataframe(settings_with_empty_working_and_output):
    assert count_files_in_folder(settings_with_empty_working_and_output.input_data_folder) == 3
    assert count_files_in_folder(settings_with_empty_working_and_output.working_data_folder) == 0
    assert count_files_in_folder(settings_with_empty_working_and_output.output_data_folder) == 0

    df = get_input_embedding_dataframe(
        settings_with_empty_working_and_output,
        "GoogleNews-vectors-negative300.bin.gz",
        load_from_cache=True,
        save_to_cache=True,
    )

    assert len(df) == 1_000_000
    assert count_files_in_folder(settings_with_empty_working_and_output.working_data_folder) == 1


def test_load_embedded_phrases(settings_with_empty_working_and_output):
    assert count_files_in_folder(settings_with_empty_working_and_output.input_data_folder) == 3
    assert count_files_in_folder(settings_with_empty_working_and_output.working_data_folder) == 0
    assert count_files_in_folder(settings_with_empty_working_and_output.output_data_folder) == 0

    df = get_embedded_phrases(
        settings_with_empty_working_and_output,
        "phrases.csv",
        "GoogleNews-vectors-negative300.bin.gz",
        load_from_cache=True,
        save_to_cache=True,
    )

    print(df)
    assert "phrases" in df.columns
    assert "embeddings" in df.columns
    assert len(df) == 50
    assert count_files_in_folder(settings_with_empty_working_and_output.working_data_folder) == 3
