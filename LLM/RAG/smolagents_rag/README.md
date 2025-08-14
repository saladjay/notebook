## smolagents RAG 示例

- 安装依赖：
```bash
pip install -r requirements.txt
```

- 准备数据：将要检索的文本放入 `data/` 目录（支持 `.txt`）。

- 构建索引：
```bash
python ingest.py --data_dir data --index_path index.npz
```

- 运行对话：
```bash
# 需设置 HUGGINGFACEHUB_API_TOKEN 环境变量，或在 --model 指定可用的推理端点模型
python chat.py --index_path index.npz --model mistralai/Mistral-7B-Instruct-v0.3 --top_k 5
```

说明：此示例使用 sentence-transformers 构建简单向量索引，并以 `smolagents` 定义检索工具；`CodeAgent` 通过 Hugging Face Inference API 或本地可用推理端点进行回答。

