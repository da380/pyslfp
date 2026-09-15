"""
Automated dataset downloading and management for the pyslfp library.

This module fetches required datasets (e.g., ice models, shapefiles, Love numbers)
from Zenodo automatically if they are not found in the local configuration directory.
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict

import requests
import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DATADIR

# The Zenodo record the datasets are fetched from: one version of the
# concept record, so it changes whenever a new version is published.
RECORD_ID: str = "22770094"

# The zip file on the record holding each dataset. These are the names as
# they appear on the record, not derived from anything else.
DATASET_FILES: Dict[str, str] = {
    "LOVE_NUMBERS": "pyslfp_love_numbers.zip",
    "ICE7G": "pyslfp_ice7g.zip",
    "ICE6G": "pyslfp_ice6g.zip",
    "ICE5G": "pyslfp_ice5g.zip",
    "HYDRO": "pyslfp_hydrobasins_v1.zip",
    "IHO_SEAS": "pyslfp_iho_seas_v3.zip",
    "TIDE_GAUGE": "pyslfp_tide_gauge.zip",
    "IMBIE_ANT": "pyslfp_imbie_ant.zip",
    "MOUGINOT_GRL": "pyslfp_mouginot_grl.zip",
    "ETOPO": "pyslfp_etopo.zip",
}

# The folder each zip extracts to under DATADIR. The folder is whatever the
# zip was made with and bears no fixed relation to the zip's name: some carry
# the pyslfp_ prefix and some do not, so the two tables are kept separately
# rather than one being derived from the other.
FOLDER_MAP: Dict[str, str] = {
    "LOVE_NUMBERS": "pyslfp_love_numbers",
    "ICE7G": "ice7g",
    "ICE6G": "ice6g",
    "ICE5G": "ice5g",
    "HYDRO": "HydroBasins",
    "IHO_SEAS": "World_Seas_IHO_v3",
    "TIDE_GAUGE": "tide_gauge",
    "IMBIE_ANT": "ANT_Basins_IMBIE2",
    "MOUGINOT_GRL": "Greenland_Basins",
    "ETOPO": "pyslfp_etopo",
}

DATASET_URLS: Dict[str, str] = {
    key: f"https://zenodo.org/records/{RECORD_ID}/files/{name}?download=1"
    for key, name in DATASET_FILES.items()
}


def _get_robust_session() -> requests.Session:
    """
    Configures a requests Session with automatic retries for flaky connections.

    Returns:
        requests.Session: A configured session object with retry logic attached.
    """
    session = requests.Session()

    # Configure the retry strategy
    retries = Retry(
        total=5,  # Try up to 5 times
        backoff_factor=1,  # Wait 1s, 2s, 4s, 8s between retries
        status_forcelist=[500, 502, 503, 504],  # Only retry on these server errors
        allowed_methods=["GET"],
    )

    # Apply the strategy to all http/https requests made by this session
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def ensure_data(dataset_key: str, /, *, refresh: bool = False) -> Path:
    """
    Checks for the data folder. If missing, automatically downloads it from Zenodo.

    Args:
        dataset_key (str): The unique identifier for the dataset (e.g., "ICE7G").
            Must be passed positionally.
        refresh (bool): If True, the cached folder is deleted and the dataset
            downloaded again, which is how a dataset that has changed on
            Zenodo is picked up. Defaults to False.

    Returns:
        Path: The absolute path to the verified local data directory.

    Raises:
        ValueError: If the dataset_key is not recognized in the FOLDER_MAP.
    """
    if dataset_key not in FOLDER_MAP:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    target = DATADIR / FOLDER_MAP[dataset_key]

    if refresh and target.exists():
        shutil.rmtree(target)

    # If the folder doesn't exist or is empty, fetch it
    if not target.exists() or not any(target.iterdir()):
        fetch_dataset(dataset_key)

    return target


def fetch_dataset(dataset_key: str, /) -> None:
    """
    Downloads and extracts a specific dataset from Zenodo.

    Args:
        dataset_key (str): The unique identifier for the dataset to download.
            Must be passed positionally.

    Raises:
        ValueError: If there is no download URL configured for the dataset.
        requests.HTTPError: If the Zenodo server returns a bad response.
    """
    url = DATASET_URLS.get(dataset_key)
    if not url:
        raise ValueError(f"No download URL configured for dataset: {dataset_key}")

    print(f"Downloading {dataset_key} dataset to {DATADIR}...")
    zip_path = DATADIR / f"{dataset_key.lower()}_temp.zip"

    # Use the robust session instead of standard requests.get
    session = _get_robust_session()

    try:
        response = session.get(url, stream=True)
        response.raise_for_status()  # Check for 404s or other non-retryable errors

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(zip_path, "wb") as f,
            tqdm.tqdm(
                total=total_size, unit="B", unit_scale=True, desc=dataset_key
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Extracting {dataset_key}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATADIR)

    finally:
        # Always clean up the zip file even if extraction fails
        if zip_path.exists():
            zip_path.unlink()
        session.close()  # Good practice to close the session

    print(f"Successfully installed {dataset_key}.")
