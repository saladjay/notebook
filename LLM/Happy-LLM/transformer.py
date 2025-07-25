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
    def __init__(self, args, is_causal=False):
        super(MultiHeadAttention, self).__init__()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.head_dim = args.dim // args.n_heads

        # wq, wk, wv 参数矩阵，每个参数矩阵 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接
        # 其实等同拼接矩阵再内积
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)

        self.resid_dropout = nn.Dropout(args.dropout)

        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer('mask', mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz, seqlen, _ = q.size() # 获取批次大小和seq len

        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 因为QKV拆分成多头，维度为（batch size, seq len, nb_heads, head_dim）
        # 我们对seq_len 和 head_dim 进行计算， 所以我们需要transpose一下
        
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, query_len, head_dim]
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, key_len, head_dim]
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, value_len, head_dim]

        # 计算注意力
        # 计算QK^T /sqrt(d_k)，维度为（B, nh, seq_len, hd) x (B, nh, hd, seq_len) -> (B, nh, seq_len, seq_len)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask') and self.mask is not None
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算注意力权重
        scores = torch.nn.functional.softmax(scores, dim=-1).type_as(xq)
        # dropout
        scores = self.attn_dropout(scores)
        #  V x scores , (B, nh, seq_len, seq_len) x (B, nh, seq_len, head_dim) -> (B, nh, seq_len, head_dim)
        output = torch.matmul(scores, xv)  # [batch_size, n_heads, query_len, head_dim]

        # 恢复时间维度并合并头
        # 将多头的结果拼接起来，先交换维度[batch_size, n_heads, query_len, head_dim] -> [batch_size, query_len, n_heads, head_dim] -> [batch_size, query_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(torch.nn.functional.relu(self.w1(x))))



class LayerNorm(nn.Module):
    '''层归一化'''
    def __init__(self, dim: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        mean = x.mean(-1, keepdim=True) # mean: [batch_size, seq_len, 1]
        std = x.std(-1, keepdim=True)   # std: [batch_size, seq_len, 1]
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class EncoderLayer(nn.Module):
    '''Encoder层'''
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.attention_norm = LayerNorm(args.dim)
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.dim)
        self.feed_forward = MLP(args.dim, args.ff_dim, args.dropout)

    def forward(self, x):
        # pre norm
        norm_x = self.attention_norm(x)
        # self attention
        h = x + self.attention(norm_x, norm_x, norm_x)
        # pre norm
        out = h + self.feed_forward(self.fnn_norm(h))
        return out
    
class Encoder(nn.Module):
    '''Encoder'''
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __call__(self, args):
        super(DecoderLayer, self).__init__()
        # 一个decoder layer 有 三个大组成部分，self-attention, cross-attention, MLP
        self.mask_attention_norm = LayerNorm(args.dim)
        # self attention 是一个Mask attention
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        # 
        self.cross_attention_norm = LayerNorm(args.dim)
        # cross attention 是一个普通的attention
        self.cross_attention = MultiHeadAttention(args, is_causal=False)
        #
        self.fnn_norm = LayerNorm(args.dim)
        # MLP
        self.feed_forward = MLP(args)

    def forward(self, x, encoder_output):
        # pre norm
        norm_x = self.mask_attention_norm(x)
        # 掩码注意力
        x = x + self.mask_attention(norm_x, norm_x, norm_x)
        # pre norm
        norm_x = self.cross_attention_norm(x)
        # 交互注意力
        h = x + self.cross_attention(norm_x, encoder_output, encoder_output)
        # pre norm
        out = h + self.feed_forward(self.fnn_norm(h))
        return out
    
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.dim)

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return self.norm(x)
    
class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        # Dropout 概率
        self.dropout = nn.Dropout(args.dropout)

        # block size
        pe = torch.zeros(args.max_seq_len, args.dim)
        position = torch.arange(0, args.max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算theta
        div_term = torch.exp(torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd))
        # 分别sin, cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加入到Embedding上
        x = x + self.pe[:, :x.size(1)].register_grad_(False)
        return self.dropout(x)
    

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.transformer = nn.Sequential(dict(
            wte = nn.Embedding(args.vocab_size, args.dim),
            # position embedding
            wpe = PositionalEncoding(args),
            # drop
            drop = nn.Dropout(args.dropout),
            # encoder
            encoder = Encoder(args),
            # decoder
            decoder = Decoder(args))
        )
        # 最后输入线性层， 输入n_embd, 输出vocab_size
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # init weights
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层 和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    '''前向计算函数'''
    def forward(self, idx, target=None):
        # 输入为idx, 维度为 （batch size, sequence length, 1): targets 为目标序列， 用于计算Loss
        device = idx.device
        b, t = idx.size()

        assert t <= self.args.max_seq_len, "Input sequence length should be less than or equal to %d" % self.args.max_seq_len
        # 通过 embedding 层
        token_embedding = self.transformer.wte(idx)

        # 位置编码
        pos_embedding = self.transformer.wpe(token_embedding)

        # dropout
        x = self.transformer.drop(pos_embedding)

        # encoder
        encoder_output = self.transformer.encoder(x)

        # decoder
        x = self.transformer.decoder(x, encoder_output)

        if target is not None:
            logits = self.lm_head(x)
            # 计算loss
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0)
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
