import faiss
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from transformers import AutoModel, AutoTokenizer
model_name='bert-base-uncased'
tokenizer=BertTokenizer.from_pretrained(model_name)
print(type(tokenizer))
model=BertModel.from_pretrained(model_name)
model.eval()

model_name = "Qwen/Qwen3-Embedding-0.6B"  # Hugging Face模型ID
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

documents = [
    'Artificial intelligence is transforming the world.',
    "FAISS is a library for efficient similarity search.",
    "Transformers library provides state-of-the-art NLP models.",
    "RAG combines retrieval and generation for better answers.",
    "Machine learning enables computers to learn from data."
]

# text to vector
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # print(f"sentence {text}, {inputs}")
    outputs=model(**inputs)
    # print(outputs.last_hidden_state.detach().numpy().shape)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# doc to vector

document_embeddings = np.array([get_sentence_embedding(doc)[0] for doc in documents])


dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

query = "What is FAISS ?"
query_embedding = get_sentence_embedding(query)[0]

k = 2
distances, indices = index.search(np.array([query_embedding]), k)

print("查询", query)
for i, idx in enumerate(indices[0]):
    print(f"Top-{i+1} 最相似的文档： {documents[idx]} (距离：{distances[0][i]:.4f})")
