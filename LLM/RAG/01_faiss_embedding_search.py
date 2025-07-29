import faiss
import numpy as np
from transformers import BertModel, BertTokenizer
import torch

model_name='bert-base-uncased'
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name)

documents = [
    
]
