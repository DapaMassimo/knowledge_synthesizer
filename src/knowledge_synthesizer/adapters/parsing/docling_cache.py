"""Content-hash cache for parsed DoclingDocuments (in-memory + optional disk).

The in-memory layer is always active and is how the parser hands the native
DoclingDocument to the chunker within a run. The disk layer (when a directory is
configured) makes re-parsing instant across runs and can be bypassed per call
(the ``--no-cache`` path) without breaking the in-memory hand-off.
"""

from __future__ import annotations

from pathlib import Path

from docling_core.types.doc import DoclingDocument  # type: ignore[attr-defined]


class DoclingDocumentCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._memory: dict[str, DoclingDocument] = {}
        self._cache_dir = cache_dir

    def get(
        self, content_hash: str, *, use_disk: bool = True, disk_suffix: str = ""
    ) -> DoclingDocument | None:
        cached = self._memory.get(content_hash)
        if cached is not None:
            return cached
        if use_disk and self._cache_dir is not None:
            path = self._cache_dir / f"{content_hash}{disk_suffix}.json"
            if path.exists():
                document = DoclingDocument.model_validate_json(path.read_text(encoding="utf-8"))
                self._memory[content_hash] = document
                return document
        return None

    def put(
        self,
        content_hash: str,
        document: DoclingDocument,
        *,
        use_disk: bool = True,
        disk_suffix: str = "",
    ) -> None:
        self._memory[content_hash] = document
        if use_disk and self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / f"{content_hash}{disk_suffix}.json"
            path.write_text(document.model_dump_json(), encoding="utf-8")

    def delete(self, content_hash: str) -> None:
        """Drop a cached document from memory and disk (all parse variants for the hash)."""
        self._memory.pop(content_hash, None)
        if self._cache_dir is not None:
            for path in self._cache_dir.glob(f"{content_hash}*.json"):
                path.unlink(missing_ok=True)
