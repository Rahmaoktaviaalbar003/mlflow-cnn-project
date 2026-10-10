
import hashlib
import json
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_FILE = Path("dataset_hash.json")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


files = sorted(
    path for path in DATA_DIR.rglob("*")
    if path.is_file()
)

if not files:
    raise RuntimeError(
        "Tidak ditemukan file dataset di folder data."
    )

manifest = {
    str(path.relative_to(DATA_DIR)): sha256_file(path)
    for path in files
}

combined_hash = hashlib.sha256(
    json.dumps(
        manifest,
        sort_keys=True
    ).encode("utf-8")
).hexdigest()

result = {
    "dataset": "Fashion-MNIST",
    "file_count": len(files),
    "dataset_sha256": combined_hash,
    "files": manifest
}

OUTPUT_FILE.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("Jumlah file:", len(files))
print("Dataset SHA-256:", combined_hash)
print("Manifest disimpan di:", OUTPUT_FILE)
