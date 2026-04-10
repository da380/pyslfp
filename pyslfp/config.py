"""
Shared configuration constants for the pyslfp library.
"""

import os
from pathlib import Path

# Define the path to the data directory.
# We prioritize an environment variable (great for CI or dev work),
# then fallback to a hidden folder in the user's home directory.
DATADIR = Path(os.getenv("PYSLFP_DATA", Path.home() / ".pyslfp_data"))

# Ensure the directory exists so the library doesn't crash on the first import.
DATADIR.mkdir(parents=True, exist_ok=True)
