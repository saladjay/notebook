# LLM output configuration 大语言模型输出配置
## Output length 输出长度
输出长度是一个重要的配置。让大语言模型产生更多的输出token将会需要更多的计算和能源消耗，更长的回复时间和更高的消耗。限制大语言模型的输出长度不意味着模型变得更加言简意赅，而是会导致模型输出到一定长度后停止预测。如果用户需要更短的输出，工程师应该考虑调整提示词。限制大预言模型的输出长度对于类似ReAct的模型来说尤其重要，因为它们会持续的输出用户不需要的无用token。

## Sampling controls
大预言模型通常不会输出单个token，相反的，它们会在自己的词表中选择一个有可能的token。这些token会被采样，模型会根据这些token预测下一个token。temperature, top-k, top-p 三个参数可以控制模型的采样策略。

### Temperature

温度是控制token选择的随机程度。低的温数值会导致更加确定性的token选择，高的温度数值会导致更随机的token选择。0温度的时候会导致token选择变成贪心算法，总是选择概率最高的token。当温度变得越来越高是，token选择也越来越公平。

### Top-K and Top-P

