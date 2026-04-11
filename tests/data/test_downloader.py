import pytest
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
