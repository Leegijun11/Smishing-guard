import re
import sys
from pathlib import Path
from typing import Dict, List, TypedDict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.labels import LABEL_MAP
from common.settings import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR

HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
TITLE_RE = re.compile(r"^#\s+.*\n")


class Chunk(TypedDict):
    id: str
    text: str
    label: str
    label_ko: str
    section: str
    source: str


def _split_sections(markdown_text: str) -> List[Dict[str, str]]:
    """최상단 H1 제목은 제거하고 H2(##) 단위로 섹션을 나눈다."""
    text = TITLE_RE.sub("", markdown_text, count=1)

    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [{"heading": "본문", "body": text.strip()}]

    sections = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"heading": heading, "body": body})
    return sections


def _split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """섹션이 chunk_size보다 길면 줄 단위로 겹치게 나눈다."""
    if len(text) <= chunk_size:
        return [text]

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 <= chunk_size:
            current = f"{current}\n{line}".strip()
        else:
            if current:
                chunks.append(current)
            overlap_tail = current[-overlap:] if overlap else ""
            current = f"{overlap_tail}\n{line}".strip()

    if current:
        chunks.append(current)
    return chunks


def build_chunks() -> List[Chunk]:
    """data/knowledge/*.md 전부를 읽어 라벨별 청크 리스트로 변환."""
    chunks: List[Chunk] = []

    for label_key, spec in LABEL_MAP.items():
        md_path = KNOWLEDGE_DIR / f"{label_key}.md"
        if not md_path.exists():
            continue  # normal 라벨은 대처방법 문서가 없음

        raw = md_path.read_text(encoding="utf-8")
        sections = _split_sections(raw)

        chunk_idx = 0
        for section in sections:
            pieces = _split_long_text(section["body"], CHUNK_SIZE, CHUNK_OVERLAP)
            for piece in pieces:
                text_with_context = f"[{spec.ko} - {section['heading']}]\n{piece}"
                chunks.append(
                    {
                        "id": f"{label_key}_{chunk_idx}",
                        "text": text_with_context,
                        "label": label_key,
                        "label_ko": spec.ko,
                        "section": section["heading"],
                        "source": md_path.name,
                    }
                )
                chunk_idx += 1

    return chunks


if __name__ == "__main__":
    for c in build_chunks():
        print(c["id"], "|", c["section"], "|", len(c["text"]), "chars")
