import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.settings import CHROMA_COLLECTION
from vector_db.build_index import get_client, get_embedding_function


class PlaybookRetriever:
    def __init__(self):
        client = get_client()
        ef = get_embedding_function()
        self.collection = client.get_collection(CHROMA_COLLECTION, embedding_function=ef)

    def search(self, query_text: str, label: Optional[str] = None, top_k: int = 4) -> List[Dict]:
        where = {"label": label} if label else None
        result = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        return [
            {
                "text": doc,
                "section": meta["section"],
                "label": meta["label"],
                "label_ko": meta["label_ko"],
                "distance": dist,
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]


if __name__ == "__main__":
    retriever = PlaybookRetriever()
    hits = retriever.search("택배 링크를 눌렀는데 어떻게 해야하나요", label="delivery")
    for h in hits:
        print(h["section"], "-", h["distance"])