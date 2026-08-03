import re
import sys
from pathlib import Path
from typing import Dict, List, TypedDict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.labels import LABEL_MAP
from common.settings import KNOWLEDGE_DIR

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


def build_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []

    for label_key, spec in LABEL_MAP.items():
        md_path = KNOWLEDGE_DIR / f"{label_key}.md"
        if not md_path.exists():
            continue  

        raw = md_path.read_text(encoding="utf-8")
        sections = _split_sections(raw)

        for idx, section in enumerate(sections):
            text_with_context = f"[{spec.ko} - {section['heading']}]\n{section['body']}"
            chunks.append(
                {
                    "id": f"{label_key}_{idx}",
                    "text": text_with_context,
                    "label": label_key,
                    "label_ko": spec.ko,
                    "section": section["heading"],
                    "source": md_path.name,
                }
            )

    return chunks


if __name__ == "__main__":
    for c in build_chunks():
        print(c["id"], "|", c["section"], "|", len(c["text"]), "chars")
