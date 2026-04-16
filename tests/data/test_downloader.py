import pytest

import requests
from unittest.mock import patch, MagicMock

# Updated import paths
import pyslfp.data.downloader as dl
import pyslfp.data.config as cfg


@pytest.fixture
def mock_datadir(tmp_path, monkeypatch):
    """Overrides DATADIR to use a temporary directory for testing."""
    monkeypatch.setattr(cfg, "DATADIR", tmp_path)
    monkeypatch.setattr(dl, "DATADIR", tmp_path)
    return tmp_path


@patch("pyslfp.data.downloader._get_robust_session")
@patch("pyslfp.data.downloader.zipfile.ZipFile")
def test_ensure_data_triggers_download(mock_zipfile, mock_session, mock_datadir):
    """Test that missing data triggers a network fetch and extraction."""
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "1024"}
    mock_response.iter_content.return_value = [b"fake", b"zip", b"data"]

    mock_session_instance = mock_session.return_value
    mock_session_instance.get.return_value = mock_response

    target_path = dl.ensure_data("LOVE_NUMBERS")

    assert target_path == mock_datadir / "love_numbers"
    expected_url = dl.DATASET_URLS["LOVE_NUMBERS"]
    mock_session_instance.get.assert_called_once_with(expected_url, stream=True)

    mock_zipfile_instance = mock_zipfile.return_value.__enter__.return_value
    mock_zipfile_instance.extractall.assert_called_once_with(mock_datadir)


def test_ensure_data_skips_download_if_present(mock_datadir, monkeypatch):
    """Test that existing data bypasses the download logic."""
    dataset_key = "LOVE_NUMBERS"
    folder_name = dl.FOLDER_MAP[dataset_key]
    target_dir = mock_datadir / folder_name
    target_dir.mkdir()
    (target_dir / "dummy.nc").write_text("fake netcdf data")

    fetch_called = False

    def mock_fetch(key):
        nonlocal fetch_called
        fetch_called = True

    monkeypatch.setattr(dl, "fetch_dataset", mock_fetch)

    result_path = dl.ensure_data(dataset_key)
    assert result_path == target_dir
    assert fetch_called is False, "Tried to fetch existing data!"


def test_unknown_dataset_key_raises_error():
    with pytest.raises(ValueError, match="Unknown dataset key: FAKE_KEY"):
        dl.ensure_data("FAKE_KEY")


def test_ensure_data_triggers_download_if_folder_empty(mock_datadir, monkeypatch):
    """Test that an existing but EMPTY data folder still triggers a fetch."""
    dataset_key = "LOVE_NUMBERS"
    folder_name = dl.FOLDER_MAP[dataset_key]
    target_dir = mock_datadir / folder_name

    # Create the directory, but put NO files in it
    target_dir.mkdir()

    fetch_called = False

    def mock_fetch(key):
        nonlocal fetch_called
        fetch_called = True

    monkeypatch.setattr(dl, "fetch_dataset", mock_fetch)

    dl.ensure_data(dataset_key)
    assert fetch_called is True, "Fetch was not called for an empty directory!"


@patch("pyslfp.data.downloader._get_robust_session")
def test_fetch_dataset_raises_http_error_on_bad_status(mock_session, mock_datadir):
    """Test that Zenodo server errors bubble up correctly."""
    mock_response = MagicMock()
    # Force raise_for_status to raise a mock HTTPError
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")

    mock_session_instance = mock_session.return_value
    mock_session_instance.get.return_value = mock_response

    with pytest.raises(requests.HTTPError, match="404 Client Error"):
        dl.fetch_dataset("LOVE_NUMBERS")


@patch("pyslfp.data.downloader._get_robust_session")
@patch("pyslfp.data.downloader.zipfile.ZipFile")
def test_fetch_dataset_cleans_up_zip_on_extraction_failure(
    mock_zipfile, mock_session, mock_datadir
):
    """Test the finally block to ensure temp zip files are deleted if extraction crashes."""
    dataset_key = "LOVE_NUMBERS"
    zip_path = mock_datadir / f"{dataset_key.lower()}_temp.zip"

    mock_response = MagicMock()
    mock_response.headers = {"content-length": "1024"}
    mock_response.iter_content.return_value = [b"fake", b"data"]

    mock_session_instance = mock_session.return_value
    mock_session_instance.get.return_value = mock_response

    # Force the extraction step to crash
    mock_zipfile_instance = mock_zipfile.return_value.__enter__.return_value
    mock_zipfile_instance.extractall.side_effect = Exception("Simulated disk error")

    with pytest.raises(Exception, match="Simulated disk error"):
        dl.fetch_dataset(dataset_key)

    # The crucial assertion: the zip file should NOT exist because the 'finally' block ran
    assert (
        not zip_path.exists()
    ), "The temporary zip file was not cleaned up after an error!"
