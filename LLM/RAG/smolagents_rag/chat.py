import argparse
import sys
from typing import List
import numpy as np
from smolagents import Tool, CodeAgent, InferenceClientModel
from typing import Optional
import os
from http import HTTPStatus


class LocalRetrieverTool(Tool):
    name = "local_retriever"
    description = (
        "从本地向量索引中检索与查询最相关的段落。"
        "输入为字符串查询，输出为拼接的上下文，供模型回答问题。"
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "检索查询字符串",
        }
    }
    output_type = "string"

    def __init__(self, embeddings: np.ndarray, texts: List[str], sources: List[str], top_k: int = 5):
        super().__init__()
        self.embeddings = embeddings  # (N, D) L2-normalized
        self.texts = texts
        self.sources = sources
        self.top_k = top_k

    def _embed_query(self, query: str, embed_model_tag: str) -> np.ndarray:
        if embed_model_tag.startswith("dashscope:"):
            model_id = embed_model_tag.split(":", 1)[1]
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY is not set for DashScope embeddings.")
            try:
                from dashscope import TextEmbedding  # type: ignore
            except Exception:
                try:
                    from dashscope.api_resources.embedding import Embedding as TextEmbedding  # type: ignore
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError("dashscope is not installed. Please `pip install dashscope`.") from exc
            os.environ.setdefault("DASHSCOPE_API_KEY", api_key)
            result = TextEmbedding.call(model=model_id, input=[query])
            status_code = getattr(result, "status_code", None)
            if status_code is not None and status_code != HTTPStatus.OK:
                raise RuntimeError(f"DashScope embedding failed with status {status_code}: {getattr(result, 'message', '')}")
            output = getattr(result, "output", None)
            if not output or "embeddings" not in output:
                raise RuntimeError("DashScope embedding response missing 'embeddings'.")
            vec = np.asarray(output["embeddings"][0]["embedding"], dtype=np.float32)[None, :]
            return vec
        else:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(embed_model_tag)
            return encoder.encode([query], convert_to_numpy=True).astype(np.float32)

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        # Read the embed model tag stored inside the index if available via attribute
        embed_model_tag: Optional[str] = getattr(self, "embed_model", None)  # type: ignore[attr-defined]
        if not embed_model_tag:
            embed_model_tag = "sentence-transformers/all-MiniLM-L6-v2"
        q = self._embed_query(query, embed_model_tag)
        # L2 normalize
        q_norm = q / max(np.linalg.norm(q, axis=1, keepdims=True), np.array([[1.0]], dtype=np.float32))
        scores = (self.embeddings @ q_norm.T).squeeze(-1)  # cosine similarity since both normalized
        top_k = min(self.top_k, len(self.texts))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        context_blocks: List[str] = []
        for rank, i in enumerate(idx):
            context_blocks.append(f"[#{rank+1} | {self.sources[i]}]\n{self.texts[i]}")
        return "\n\n".join(context_blocks)


def load_index(index_path: str):
    data = np.load(index_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    texts = data["texts"].tolist()
    sources = data["sources"].tolist()
    embed_model = data.get("embed_model", None)
    if embed_model is not None:
        try:
            embed_model = str(embed_model)
        except Exception:
            embed_model = None
    return embeddings, texts, sources, embed_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a smolagents RAG agent")
    parser.add_argument("--index_path", type=str, required=True, help="Path to .npz index from ingest.py")
    parser.add_argument("--model", type=str, required=True, help="HF Inference model id, e.g. mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k contexts to retrieve")
    parser.add_argument("--temperature", type=float, default=0.2, help="Decoding temperature for model (if supported)")
    args = parser.parse_args()

    embeddings, texts, sources, embed_model = load_index(args.index_path)

    retriever = LocalRetrieverTool(embeddings=embeddings, texts=texts, sources=sources, top_k=args.top_k)
    # Attach embed_model tag so the tool uses the same backend for query embedding
    setattr(retriever, "embed_model", embed_model or "sentence-transformers/all-MiniLM-L6-v2")

    system_prompt = (
        "你是一个可靠的中文问答助手，擅长使用检索到的上下文进行准确、引用规范的回答。\n"
        "流程：\n"
        "1) 先调用 local_retriever 获取相关上下文。\n"
        "2) 基于上下文回答用户问题，必要时引用来源（文件名和chunk）。\n"
        "3) 如果上下文不足以回答，请明确说明，并指出需要的额外信息。\n"
    )

    model = InferenceClientModel(model_id=args.model, temperature=args.temperature)

    agent = CodeAgent(tools=[retriever], model=model, system_prompt=system_prompt)

    print("[OK] RAG agent ready. Type 'exit' to quit.")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q", "bye"}:
            print("Bye")
            return
        try:
            answer = agent.run(query)
            print(f"\nAssistant: {answer}\n")
        except Exception as exc:
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()

