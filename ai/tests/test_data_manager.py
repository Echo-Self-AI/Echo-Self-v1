from pathlib import Path
import pytest

from echoself_ai.data_manager import (
    DatasetManifest,
    ShardedTextDataset,
    TextCleaner,
    compute_sha256,
    create_manifest_for_corpus,
)


def test_text_cleaner_normalization():
    dirty = "hello\r\nworld\r"
    cleaned = TextCleaner.normalize_text(dirty)
    assert cleaned == "hello\nworld\n"


def test_text_cleaner_deduplication():
    text = "line 1\nline 1\nline 2\nline 2\nline 2\nline 3"
    deduped = TextCleaner.deduplicate_lines(text)
    assert deduped == "line 1\nline 2\nline 3"


def test_manifest_creation_and_checksum(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello echoself\ntraining data\n", encoding="utf-8")

    manifest = create_manifest_for_corpus(
        corpus_path=sample_file,
        name="test_corpus",
        version="0.1.0",
    )

    assert manifest.num_characters == 29
    assert manifest.num_lines == 2
    assert len(manifest.sha256) == 64
    assert manifest.sha256 == compute_sha256(sample_file)

    # Test serialization
    json_path = tmp_path / "manifest.json"
    json_path.write_text(manifest.to_json(), encoding="utf-8")

    loaded = DatasetManifest.from_file(json_path)
    assert loaded.name == "test_corpus"
    assert loaded.sha256 == manifest.sha256


def test_sharded_text_dataset_iteration(tmp_path: Path):
    shard1 = tmp_path / "shard1.txt"
    shard1.write_text("AAAA" * 10, encoding="utf-8")
    shard2 = tmp_path / "shard2.txt"
    shard2.write_text("BBBB" * 10, encoding="utf-8")

    dataset = ShardedTextDataset([shard1, shard2], chunk_chars=20)
    chunks = list(dataset)

    assert len(chunks) == 4  # 40 chars / 20 = 2 chunks each
    assert "".join(chunks) == ("AAAA" * 10) + ("BBBB" * 10)


def test_sharded_text_dataset_empty_files_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        ShardedTextDataset([])

    with pytest.raises(ValueError, match="chunk_chars"):
        ShardedTextDataset([Path("any.txt")], chunk_chars=0)


def test_speech_dataset_manifest_roundtrip(tmp_path: Path):
    from echoself_ai.data_manager import SpeechDatasetManifest

    manifest = SpeechDatasetManifest(
        name="test_speech",
        version="1.0",
        source="http://example.com/audio",
        license="CC0",
        num_clips=5,
        total_bytes=1024,
        sample_rate=16000,
        description="Test speech dataset",
    )
    json_path = tmp_path / "speech.manifest.json"
    json_path.write_text(manifest.to_json(), encoding="utf-8")

    loaded = SpeechDatasetManifest.from_file(json_path)
    assert loaded.name == "test_speech"
    assert loaded.num_clips == 5
    assert loaded.sample_rate == 16000


def test_fetch_ljspeech_subset_local(tmp_path: Path):
    from echoself_ai.data_manager import fetch_ljspeech_subset

    # Test with simulated local metadata
    target_dir = tmp_path / "ljspeech"
    target_dir.mkdir(parents=True)
    meta_path = target_dir / "metadata.csv"
    meta_path.write_text("LJ001-0001|Hello world|Hello world\n", encoding="utf-8")
    wavs_dir = target_dir / "wavs"
    wavs_dir.mkdir()
    (wavs_dir / "LJ001-0001.wav").write_bytes(b"RIFF" + b"\x00" * 2000)

    returned_meta, returned_wavs, manifest = fetch_ljspeech_subset(target_dir, max_clips=1)
    assert returned_meta.exists()
    assert returned_wavs.exists()
    assert manifest.num_clips == 1
    assert manifest.license.startswith("Public Domain")

