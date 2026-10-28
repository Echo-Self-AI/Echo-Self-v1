from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Optional
import unicodedata


@dataclass
class DatasetManifest:
    """Metadata, license, and provenance manifest for a training dataset."""

    name: str
    version: str
    source: str
    license: str
    num_characters: int
    num_lines: int
    sha256: str
    description: str
    is_synthetic: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


class TextCleaner:
    """Preprocesses raw text with Unicode normalization and deduplication."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Apply Unicode NFC normalization and clean stray non-printable characters."""
        # NFC normalizes composed characters
        normalized = unicodedata.normalize("NFC", text)
        # Strip carriage returns, standardize to \n
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        return normalized

    @staticmethod
    def deduplicate_lines(text: str) -> str:
        """Remove consecutive identical lines while preserving overall document structure."""
        lines = text.split("\n")
        deduped: list[str] = []
        for line in lines:
            if not deduped or line != deduped[-1]:
                deduped.append(line)
        return "\n".join(deduped)


def compute_sha256(path: str | Path) -> str:
    """Compute sha256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_manifest_for_corpus(
    corpus_path: str | Path,
    name: str,
    version: str = "0.1.0",
    source: str = "Local EchoSelf synthetic corpus",
    license_type: str = "Apache-2.0",
    description: str = "Baseline synthetic smoke testing corpus",
) -> DatasetManifest:
    """Analyze a corpus on disk and generate its DatasetManifest."""
    path = Path(corpus_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    return DatasetManifest(
        name=name,
        version=version,
        source=source,
        license=license_type,
        num_characters=len(text),
        num_lines=len(lines),
        sha256=compute_sha256(path),
        description=description,
        is_synthetic=True,
    )


class ShardedTextDataset:
    """Memory-efficient streaming iterator over text files in fixed-size chunks."""

    def __init__(
        self,
        file_paths: list[Path],
        chunk_chars: int = 1024,
    ) -> None:
        if not file_paths:
            raise ValueError("file_paths cannot be empty")
        if chunk_chars < 1:
            raise ValueError(f"chunk_chars must be >= 1, got {chunk_chars}")

        self.file_paths = [Path(p) for p in file_paths]
        self.chunk_chars = chunk_chars

    def __iter__(self) -> Iterator[str]:
        """Yield text chunks across all specified shards sequentially."""
        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Shard file not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                while chunk := f.read(self.chunk_chars):
                    yield chunk


DATASET_REGISTRY = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "tinyshakespeare.txt",
        "license": "Public Domain / Project Gutenberg",
        "description": "Public domain extract of William Shakespeare works for language modeling",
        "expected_min_bytes": 1000000,
    }
}


def fetch_and_prepare_dataset(
    name: str = "tinyshakespeare",
    target_dir: str | Path = "data/raw",
) -> tuple[Path, DatasetManifest]:
    """Download, clean, and generate manifest for a registered public domain dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")

    info = DATASET_REGISTRY[name]
    target_path = Path(target_dir) / info["filename"]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.is_file() or target_path.stat().st_size < info["expected_min_bytes"]:
        import urllib.request
        with urllib.request.urlopen(info["url"], timeout=30) as resp:
            raw_text = resp.read().decode("utf-8")

        cleaned_text = TextCleaner.normalize_text(raw_text)
        target_path.write_text(cleaned_text, encoding="utf-8")

    manifest = create_manifest_for_corpus(
        corpus_path=target_path,
        name=name,
        source=info["url"],
        license_type=info["license"],
        description=info["description"],
    )
    manifest_path = target_path.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")

    return target_path, manifest

