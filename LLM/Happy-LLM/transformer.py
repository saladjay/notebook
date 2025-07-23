import torch
import math
import torch.nn as nn
MAX_SEQ_LEN = 512  # 最大序列长度

'''attention fucntion'''
def attention(q, k, v, drop_out=None, mask=None):
    '''
    args:
        q: query tensor with shape [batch_size, query_len, d_model]
        k: key tensor with shape [batch_size, key_len, d_model]
        v: value tensor with shape [batch_size, value_len, d_model]
    '''
    # 获取查询矩阵的维度
    d_k = q.size(-1)
    # 计算q和k的内积并且除以根号d_k
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # [batch_size, query_len, key_len]
    if mask is not None:
        mask = torch.full((1, MAX_SEQ_LEN, MAX_SEQ_LEN), float('-inf'), device=scores.device)
        mask = mask.triu(diagonal=1)
        scores = scores + mask[:, :scores.size(1), :scores.size(2)]


    p_atten = torch.softmax(scores, dim=-1)
    if drop_out is not None:
        p_atten = drop_out(p_atten)
    # 计算注意力权重矩阵
    output = torch.matmul(p_atten, v)
    return output, p_atten

class ModelArgs:
    def __init__(self):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        super(MultiHeadAttention, self).__init__()
        