Q1:大模型里应用强化学习时，状态，动作空间，动作都是什么

A1: 状态是目前已生成的token，动作空间是下一步的词表，动作是生成token



Q2: RLHF训练时，Reward Model和LLM是同时训练还是先后训练，instruct GPT论文里是如何训练RM的

A2:RM先于LLM训练，instruct GPT论文里是将一个llm最后一层改成输出分数的形式，构造了一个RM模型，输入语句训练分数的方式。



Q3:训练RM时，无论是instruct GPT还是DPO， 为什么loss里有log和sigmod函数？ 直接用reward相减不行吗？

A3：$sigmod$函数将数值映射到$(0,1)$区间，表示概率。对数损失是实现最大似然估计的标准方法，他能给出一个良好的数学性质例如凸性，利于梯度下降，并且对错误预测施加了强烈的惩罚。



Q4:RLHF(指openai instruct GPT论文中)，训练LLM的损失函数是什么？

A4: 优势函数$A^{\pi_ {\theta _k}}(s,a)$乘于策略优势$\frac{\pi_ \theta(a|s)}{\pi_ {\theta_k}(a|s)}$，策略优势$\frac{\pi_ \theta(a|s)}{\pi_ {\theta_k}(a|s)}$在$(1-\epsilon, 1+\epsilon)$截断后乘于优势函数$A^{\pi_ {\theta _k}}(s,a)$, 上诉两者的最小值。



Q5:了解RLHF-PPO吗，里面需要训练几个模型，加载几个模型

A5：训练两个RM和LLM，需要加载4个模型，还需要存储一个LLM旧模型



Q6:RLHF-PPO里，reward的设计是什么，绝对优势估计是什么

A6: