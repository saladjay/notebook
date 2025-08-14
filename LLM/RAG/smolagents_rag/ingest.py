import argparse
import os
import glob
from typing import List, Tuple
import numpy as np
from http import HTTPStatus
from typing import Iterable
from typing import Optional
from sentence_transformers import SentenceTransformer


def read_text_files(data_dir: str) -> List[Tuple[str, str]]:
    file_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
    contents: List[Tuple[str, str]] = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                contents.append((path, f.read()))
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
    return contents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start: int = 0
    text_length: int = len(text)
    while start < text_length:
        end: int = min(start + chunk_size, text_length)
        chunk: str = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    total = len(items)
    for start in range(0, total, batch_size):
        yield items[start:start + batch_size]


def embed_with_dashscope(texts: List[str], model_id: str, api_key: Optional[str] = None, batch_size: int = 128) -> np.ndarray:
    try:
        # Lazy imports to avoid hard dependency when not used
        from dashscope import TextEmbedding  # type: ignore
    except Exception:
        try:
            from dashscope.api_resources.embedding import Embedding as TextEmbedding  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("dashscope is not installed. Please `pip install dashscope`.") from exc

    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set. Please export it to use DashScope embeddings.")

    # DashScope SDK reads api key from env var `DASHSCOPE_API_KEY`
    os.environ.setdefault("DASHSCOPE_API_KEY", api_key)

    vectors: List[np.ndarray] = []
    for batch in _batch_iter(texts, batch_size=batch_size):
        result = TextEmbedding.call(model=model_id, input=batch)
        # Some versions expose status code as `status_code`, and output embeddings at `output["embeddings"]`
        status_code = getattr(result, "status_code", None)
        if status_code is not None and status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope embedding failed with status {status_code}: {getattr(result, 'message', '')}")
        output = getattr(result, "output", None)
        if not output or "embeddings" not in output:
            raise RuntimeError("DashScope embedding response missing 'embeddings'.")
        for item in output["embeddings"]:
            emb = np.asarray(item.get("embedding"), dtype=np.float32)
            vectors.append(emb)

    if not vectors:
        raise RuntimeError("DashScope returned no embeddings.")
    return np.vstack(vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple embedding index for RAG")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .txt files")
    parser.add_argument("--index_path", type=str, required=True, help="Output .npz index path")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model. Use 'dashscope:<model_id>' for Aliyun DashScope, e.g. 'dashscope:text-embedding-v3'. Otherwise a SentenceTransformers model id.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Character chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Character chunk overlap")
    args = parser.parse_args()

    use_dashscope: bool = args.embed_model.startswith("dashscope:")
    dashscope_model_id: Optional[str] = None
    st_model: Optional[SentenceTransformer] = None

    if use_dashscope:
        dashscope_model_id = args.embed_model.split(":", 1)[1].strip()
        if not dashscope_model_id:
            raise SystemExit("Invalid dashscope model id. Example: --embed_model dashscope:text-embedding-v3")
        print(f"[INFO] Using DashScope embedding model: {dashscope_model_id}")
    else:
        print(f"[INFO] Loading embedding model: {args.embed_model}")
        st_model = SentenceTransformer(args.embed_model)

    print(f"[INFO] Scanning txt files under: {args.data_dir}")
    raw_docs = read_text_files(args.data_dir)
    if not raw_docs:
        raise SystemExit("No .txt files found in data_dir")

    texts: List[str] = []
    sources: List[str] = []
    for path, content in raw_docs:
        chunks = chunk_text(content, args.chunk_size, args.chunk_overlap)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            sources.append(f"{path}#chunk={i}")

    print(f"[INFO] Total chunks: {len(texts)}")
    if use_dashscope:
        embeddings = embed_with_dashscope(texts, model_id=dashscope_model_id)  # type: ignore[arg-type]
    else:
        embeddings = st_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)  # type: ignore[union-attr]
    embeddings = l2_normalize(embeddings.astype(np.float32))

    np.savez(
        args.index_path,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        sources=np.array(sources, dtype=object),
        embed_model=args.embed_model,
    )
    print(f"[OK] Index written to: {args.index_path}")


if __name__ == "__main__":
    main()

