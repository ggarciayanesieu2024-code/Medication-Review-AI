"""
Download the official AEMPS prescription nomenclator and unpack it.

The nomenclator is the catalogue of every medicine authorised and marketed in Spain.
It is published by the Spanish Agency for Medicines and Health Products (AEMPS).

What this script does:
1. Downloads one compressed file (a ZIP) from the AEMPS website.
2. Unpacks it into data/raw/nomenclator/.
3. Lists the files it contains, so we can see their format before parsing them.

Why this comes first:
- It is one official download with every medicine, so we know exact names,
  registration numbers and ATC codes before we query anything else.
- Once downloaded, the project works without internet access, which protects the demo.

Run from the project root:
    python -m src.Data.download_nomenclator
"""

import zipfile
from pathlib import Path

import requests


# Official AEMPS download address. It is published over plain HTTP.
NOMENCLATOR_URL = "http://listadomedicamentos.aemps.gob.es/prescripcion.zip"

# Where the raw download and its contents are stored.
RAW_DIR = Path("data") / "raw" / "nomenclator"
ZIP_PATH = RAW_DIR / "prescripcion.zip"

# The file is large, so we allow more time than for a normal API call.
REQUEST_TIMEOUT_SECONDS = 120

# We read and write the file in 1 MB pieces. This keeps memory use low,
# because we never load the whole file into memory at once.
CHUNK_SIZE_BYTES = 1024 * 1024


def download_zip(url: str, destination: Path) -> None:
    """Download a file from url and write it to destination, piece by piece.

    stream=True tells requests not to load the full response into memory.
    raise_for_status() stops the script if the server returns an error,
    so we never save an error page as if it were the ZIP file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(destination, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                output_file.write(chunk)

    size_mb = destination.stat().st_size / 1_000_000
    print(f"Downloaded {destination} ({size_mb:.1f} MB)")


def unpack_zip(zip_path: Path, destination_dir: Path) -> list[str]:
    """Extract every file from the ZIP into destination_dir and return their names.

    Returning the names lets us print them and check the format before we
    write any parsing code.
    """
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_dir)
        return archive.namelist()


def main() -> None:
    # Skip the download if the ZIP is already on disk, so re-running the script
    # does not download a large file again. If the ZIP is corrupted, delete it
    # and run the script once more.
    if ZIP_PATH.exists():
        print("ZIP already on disk, skipping download.")
    else:
        download_zip(NOMENCLATOR_URL, ZIP_PATH)

    file_names = unpack_zip(ZIP_PATH, RAW_DIR)

    print(f"\nThe ZIP contains {len(file_names)} files:")
    for name in file_names:
        print("  -", name)


if __name__ == "__main__":
    main()