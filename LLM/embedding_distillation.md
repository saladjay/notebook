# Embedding Distallation

### embedding 训练数据结构

1. 三元组格式

   ```json
   {
   "query":"查询文本",
   "positive":"正样本文本",
   "negative":"负样本文本"
   }
   ```

   说明

   * query 查询文本，作为比较的基准点
   * positive 与query相似的正样本
   * negative 与query不相似的负样本

2. 文本对格式

-- 待续

### 蒸馏数据构造

1. 读取query，positive, negative的样本，并且构造一个unique_texts记录唯一的语句，同时构造一个[query, positive, negative]的三元组

   ```python
   unique_texts = set()
   triplets = []
   with open(input_path, 'r', encoding='utf-8') as f:
       for line in f:
           data = json.loads(line)
           query = data.get('query')
           positives = data.get('positive')
           negatives = data.get('negative')
   
           if not all([query, positives, negatives]):
               continue
   
           unique_texts.add(query)
           # DistillKLDivLoss 可以处理多个负样本，但为保持与原逻辑一致，此处仍使用笛卡尔积
           # 更高效的做法是每个 query 对应1个 positive 和所有 negatives
           for pos_item, neg_item in product(positives, negatives):
               unique_texts.add(pos_item)
               unique_texts.add(neg_item)
               triplets.append({'query': query, 'positive': pos_item, 'negative': neg_item})
   ```

   

2. 利用llm可以批量推理的特性，一次性推理所有句子的embedding

   ```python
   input_texts = list(unique_texts)
   all_embeddings = []
   # 使用 model.embed() 批量处理
   outputs = model.embed(input_texts)
   all_embeddings = [torch.tensor(o.outputs.embedding, dtype=torch.float32) for o in outputs]
   logging.info("向量生成完毕。")
   # 3. 创建 文本 -> 向量 的映射字典
   text_to_embedding = {text: emb for text, emb in zip(input_texts, all_embeddings)}
   ```

   

3. 生成蒸馏数据，计算negative和positive与query的embedding相似度，并且把三元组更新一下，记录到文件里。

   ```python
   for triplet in tqdm(triplets, desc=f"为 {os.path.basename(input_path)} 计算标签"):
       q_text = triplet['query']
       p_text = triplet['positive']
       n_text = triplet['negative']
   
       emb_q = text_to_embedding.get(q_text)
       emb_p = text_to_embedding.get(p_text)
       emb_n = text_to_embedding.get(n_text)
   
       if emb_q is None or emb_p is None or emb_n is None:
           logging.warning(f"跳过一个无法找到全部向量的三元组: {triplet}")
           continue
   
       sim_pos = similarity(emb_q, emb_p)
       sim_neg = similarity(emb_q, emb_n)
   
       # 为 DistillKLDivLoss 创建标签：[positive_score, negative_score]
       label = [sim_pos, sim_neg]
   
       record = {
           "query": q_text,
           "positive": p_text,
           "negative": n_text,
           "label": label  # 标签格式为分数列表
       }
       f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
   ```

   



### 使用DistillKLDivLoss进行蒸馏训练

#### DistillKLDivLoss

* 核心原理：基于KL散度来度量两个概率分布之间的差异

* 数学定义：$P(i)$是老师模型输出的概率分布，$Q(i)$是学生模型输出的概率分布
  $$
  D_{KL}(P||Q)=\sum_iP(i)log\frac{P(i)}{Q(i)}
  $$
  在知识蒸馏中，通常使用带温度参数的软化版本，使用$softmax$产生概率分布
  $$
  L_{distill}=T^2 \cdot D_{KL}(softmax(z_s/T)||softmax(z_t/T))
  $$
  
* 蒸馏版本的KL散度会让学生模型学习老师模型产生的软标签，而不是非对即错的硬标签，让学生模型学习更加精细，信息量更大的监督信号。
* 温度参数$T$的作用
  * 平滑分布：当$T>1$时，$softmax$的输出会更加平滑，指数函数在$x$靠近0的位置会更加平滑。T越大，意味着老师模型对于不同文档的相似度分数差异会越小，概率分布没有那么尖锐。
  * 提供更多信息：平滑后的分布揭示了不同文档之间的更加细微的相关性差异，为学生模型提供更加丰富的梯度信息
* 梯度稳定技巧：损失函数前面$T^2$是一个重要的而梯度缩放因子。这是喂了补偿因使用的温度参数而导致的梯度缩放效应。当$T$较大的时候，$Softmax$产生的概率分布更平缓，会导致梯度变小。乘于$T^{2}$可以确保有效梯度大小。

#### 信息熵，交叉熵，KL离散，softmax

| 概念    | 核心公式                                      | 接近的核心问题                                       | 本质/作用                                                    |
| ------- | --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| 信息熵  | $H(p)=-\sum p(x)\cdot \log p(x)$              | 衡量一个事件或者系统的固有的不确定性或者混乱程度     | 不确定性的衡量尺度                                           |
| 交叉熵  | $H(p,q)=-\sum p(x)\cdot \log q(x)$            | 衡量用分布q来表示真实分布p所需的平均信息量           | 实际应用中**损失函数**的常用形式。值越小，q 与 p 越接近      |
| KL散度  | $D_{kd}(p||q)=\sum p(x)\cdot \log(p(x)/q(x))$ | 衡量两个概率分布 **p 和 q 之间的差异程度**           | “分布距离”的度量。非对称，用于衡量用 q 拟合 p 的**信息损失**。 |
| softmax | $Softmax(z)_i=\frac{e^{z_i}}{\sum e^{z_j}}$   | 将一组任意实数（如模型原始输出）**转换为概率分布**。 | 一个**转换器/缩放器**，将数值映射为概率，不直接衡量差异。    |


