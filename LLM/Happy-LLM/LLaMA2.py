from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
import torch
from typing import Optional, Tuple
import math
class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self, 
            dim: int = 768, # 模型维度
            n_layers: int = 12, # Transformer层数
            n_heads: int = 16, # Multi-Head Attention的头数
            n_kv_heads: int = 8, # Key-Value Attention的头数
            vocab_size: int = 6144, # 词表大小
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5, # 归一化的epsilon值
            max_seq_len: int = 512, # 最大序列长度
            dropout: float = 0.0, # Dropout值
            flash_attn: bool = True, # 是否使用Flash-Attention
            **kwargs
        ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        # eps 是为了防止除以0的情况
        self.eps = eps
        # weight是一个可学习的参数，初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算x的平方的均值
        # torch.rsqrt()计算平方根的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转化成float类型，然后进行RMSNorm，最后再转会原来的数据类型
        # 最后乘于weight, 这里RMSNorm的weight是一个可学习的参数
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果如果重复次数为1， 则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :] # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 将新添加的维度扩展到n_rep大小，实现重复效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # reshape, 合并键值对头的数量和重复次数的维度
    )

# 注意：此处的dim应为 dim // n_head, 因为我们是对每个head进行选中嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float() 生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim, 再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，步长为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
    
def reshape_for_broadcast(freqs_cls: torch.Tensor, x: torch.Tensor):
    # 获取x的维度
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cls.shape == (x.shape[1], x.shape[-1])

    # 
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cls.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor, 
        xk: torch.Tensor, 
        freqs_cos: torch.Tensor, 
        freqs_sin: torch.Tensor, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将查询和键张量展平为二维矩阵，然后将其拆分为实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 重新拼接为复数张量
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(3)
    xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        # 模型并行处理大小，默认为1
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于剪枝头数除以模型并行处理大小。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # 定义输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout的概率
        self.dropout = args.dropout

        # 检查是否使用Flash Attention，需要pytorch 2.0 以上版本
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度
        bsz, seqlen, _ = x.shape

        # 计算 QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对键和值进行扩展以适应重复次数
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention, 选择实现方式。
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = torch.nn.functional.softmax(scores, dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 回复时间维度并合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义第一层线性变换， 从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换， 从隐藏维度到输出维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换， 从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))
    

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MLP(dim = args.dim, hidden_dim = args.hidden_dim, multiple_of = args.multiple_of, dropout = args.dropout)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention_norm(self.attention(self.attention_norm(x), freqs_cos, freqs_sin))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    

class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # 定义词嵌入层
        self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropoutc层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder 层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim ,args.vocab_size, bias=False)

        self.token_embeddings.weight = self.output.weight

        # 预先计算相对位置的嵌入频率
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // self.args.n_heads, args.max_seq_len)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(self.n_layers * 2))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **keyargs) -> torch.Tensor:
        '''
        - tokens: 输入token张量
        - targets: 目标token张量
        - kv_cache: bool, 是否使用键值缓存
        - keyargs: 其他关键字参数。
        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        '''

        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        _bsz, seqlen = tokens.shape
        h = self.token_embeddings(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__("logits", logtis)
        self.OUT.__setitem__("last_loss", self.last_loss)

        return self.OUT

    @torch.infernce_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列idx(batch_size, seq_len)的长整型张量，通过多次生成新的token来完成序列。
        在model.eval()模式下运行。
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1)<=self.args.max_seq_len else idx[:, -self.args.max_seq_len]

            # 前向传播获取序列中最后一个位置的logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出

            if temperature == 0.0:
                _, idx_nex = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits<v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next == stop_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]
