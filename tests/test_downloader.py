import pytest
from unittest.mock import patch, MagicMock
import pyslfp.downloader as dl


@pytest.fixture
def mock_datadir(tmp_path, monkeypatch):
    """
    Overrides the DATADIR in the downloader module to use a temporary directory
    that gets automatically cleaned up after the test.
    """
    monkeypatch.setattr(dl, "DATADIR", tmp_path)
    return tmp_path


@patch("pyslfp.downloader._get_robust_session")
@patch("pyslfp.downloader.zipfile.ZipFile")
def test_ensure_data_triggers_download(mock_zipfile, mock_session, mock_datadir):
    """Test that missing data triggers a network fetch and extraction."""

    # 1. SETUP: Create a fake network response
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "1024"}
    # Simulate receiving chunks of data
    mock_response.iter_content.return_value = [b"fake", b"zip", b"data"]

    mock_session_instance = mock_session.return_value
    mock_session_instance.get.return_value = mock_response

    # 2. EXECUTE: Ask for data that doesn't exist in the temp directory
    target_path = dl.ensure_data("LOVE_NUMBERS")

    # 3. ASSERT: Verify the logic behaved correctly
    # Did it return the expected path?
    assert target_path == mock_datadir / "love_numbers"

    # Did it try to hit the correct Zenodo URL?
    expected_url = dl.DATASET_URLS["LOVE_NUMBERS"]
    mock_session_instance.get.assert_called_once_with(expected_url, stream=True)

    # Did it attempt to extract the zip file to the correct directory?
    mock_zipfile_instance = mock_zipfile.return_value.__enter__.return_value
    mock_zipfile_instance.extractall.assert_called_once_with(mock_datadir)


def test_ensure_data_skips_download_if_present(mock_datadir, monkeypatch):
    """Test that existing data bypasses the download logic."""

    # 1. SETUP: Create a fake "already downloaded" directory
    dataset_key = "LOVE_NUMBERS"
    folder_name = dl.FOLDER_MAP[dataset_key]
    target_dir = mock_datadir / folder_name
    target_dir.mkdir()

    # Put a dummy file inside so it's not "empty"
    (target_dir / "dummy.nc").write_text("fake netcdf data")

    # Flag to check if fetch_dataset is called
    fetch_called = False

    def mock_fetch(key):
        nonlocal fetch_called
        fetch_called = True

    # Temporarily replace the fetch function to ensure it DOES NOT run
    monkeypatch.setattr(dl, "fetch_dataset", mock_fetch)

    # 2. EXECUTE
    result_path = dl.ensure_data(dataset_key)

    # 3. ASSERT
    assert result_path == target_dir
    assert (
        fetch_called is False
    ), "The downloader tried to fetch data that already existed!"


def test_unknown_dataset_key_raises_error():
    """Ensure invalid keys are caught immediately."""
    with pytest.raises(ValueError, match="Unknown dataset key: FAKE_KEY"):
        dl.ensure_data("FAKE_KEY")
