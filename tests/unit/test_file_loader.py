import hashlib
from pathlib import Path

import pytest

from knowledge_synthesizer.adapters.loaders.file_loader import _EXTENSION_MIME, FileLoader
from knowledge_synthesizer.domain.errors import SourceLoadError
from knowledge_synthesizer.domain.models import FileSource, WebSource

pytestmark = pytest.mark.unit


async def test_loads_text_file_with_hash_and_mime(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("hello world", encoding="utf-8")

    [raw] = await FileLoader().load(FileSource(path=str(path)))

    assert raw.content == b"hello world"
    assert raw.mime_type == "text/plain"
    assert raw.content_hash == hashlib.sha256(b"hello world").hexdigest()
    assert raw.source.uri == str(path)


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".pptx", ".md"])
async def test_office_and_pdf_mime_types(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"file{suffix}"
    path.write_bytes(b"binary")

    [raw] = await FileLoader().load(FileSource(path=str(path)))

    assert raw.mime_type == _EXTENSION_MIME[suffix]


async def test_unknown_extension_falls_back_to_octet_stream(tmp_path: Path) -> None:
    path = tmp_path / "file.unknownext"
    path.write_bytes(b"x")

    [raw] = await FileLoader().load(FileSource(path=str(path)))

    assert raw.mime_type == "application/octet-stream"


async def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceLoadError):
        await FileLoader().load(FileSource(path=str(tmp_path / "nope.txt")))


async def test_rejects_web_source() -> None:
    with pytest.raises(SourceLoadError):
        await FileLoader().load(WebSource(url="https://example.com"))
