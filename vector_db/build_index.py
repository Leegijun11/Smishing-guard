import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from common.settings import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL
from vector_db.chunker import build_chunks


def get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_embedding_function():
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def build_index(reset: bool = True) -> None:
    client = get_client()
    ef = get_embedding_function()

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass  

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = build_chunks()
    if not chunks:
        raise RuntimeError("청크가 하나도 생성되지 않았습니다. data/knowledge/*.md 를 확인하세요.")

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "label": c["label"],
                "label_ko": c["label_ko"],
                "section": c["section"],
                "source": c["source"],
            }
            for c in chunks
        ],
    )

    print(f"색인 완료: {len(chunks)}개 청크 -> {CHROMA_DIR} (collection={CHROMA_COLLECTION})")


if __name__ == "__main__":
    build_index()