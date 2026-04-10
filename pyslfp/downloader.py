import zipfile
import requests
import tqdm
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import DATADIR

# The unique identifier for your Zenodo record
RECORD_ID = "19494464"

# Centralized mapping of dataset keys to their local folder names
FOLDER_MAP = {
    "LOVE_NUMBERS": "love_numbers",
    "ICE7G": "ice7g",
    "ICE6G": "ice6g",
    "ICE5G": "ice5g",
    "HYDRO": "HydroBasins",
    "IHO_SEAS": "World_Seas_IHO_v3",
    "TIDE_GAUGE": "tide_gauge",
    "IMBIE_ANT": "ANT_Basins_IMBIE2",
    "MOUGINOT_GRL": "Greenland_Basins",
}

# 1. Base automated generator
DATASET_URLS = {
    key: f"https://zenodo.org/records/{RECORD_ID}/files/pyslfp_{val.lower()}.zip?download=1"
    for key, val in FOLDER_MAP.items()
}

# 2. Overrides for specific filenames on Zenodo
DATASET_URLS["HYDRO"] = (
    f"https://zenodo.org/records/{RECORD_ID}/files/pyslfp_hydrobasins_v1.zip?download=1"
)
DATASET_URLS["IHO_SEAS"] = (
    f"https://zenodo.org/records/{RECORD_ID}/files/pyslfp_iho_seas_v3.zip?download=1"
)
DATASET_URLS["IMBIE_ANT"] = (
    f"https://zenodo.org/records/{RECORD_ID}/files/pyslfp_imbie_ant.zip?download=1"
)
DATASET_URLS["MOUGINOT_GRL"] = (
    f"https://zenodo.org/records/{RECORD_ID}/files/pyslfp_mouginot_grl.zip?download=1"
)


def _get_robust_session() -> requests.Session:
    """Configures a requests Session with automatic retries for flaky connections."""
    session = requests.Session()

    # Configure the retry strategy
    retries = Retry(
        total=5,  # Try up to 5 times
        backoff_factor=1,  # Wait 1s, 2s, 4s, 8s between retries
        status_forcelist=[500, 502, 503, 504],  # Only retry on these server errors
        allowed_methods=["GET"],
    )

    # Apply the strategy to all https:// requests made by this session
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def ensure_data(dataset_key: str) -> Path:
    """
    Checks for the data folder. If missing, automatically downloads it from Zenodo.

    Returns:
        Path: The absolute path to the verified data directory.
    """
    if dataset_key not in FOLDER_MAP:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    target = DATADIR / FOLDER_MAP[dataset_key]

    # If the folder doesn't exist or is empty, fetch it
    if not target.exists() or not any(target.iterdir()):
        fetch_dataset(dataset_key)

    return target


def fetch_dataset(dataset_key: str) -> None:
    """Downloads and extracts a specific dataset from Zenodo."""
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

        with open(zip_path, "wb") as f, tqdm.tqdm(
            total=total_size, unit="B", unit_scale=True, desc=dataset_key
        ) as pbar:
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
