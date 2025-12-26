# LLM output configuration 大语言模型输出配置
## Output length 输出长度
输出长度是一个重要的配置。让大语言模型产生更多的输出token将会需要更多的计算和能源消耗，更长的回复时间和更高的消耗。限制大语言模型的输出长度不意味着模型变得更加言简意赅，而是会导致模型输出到一定长度后停止预测。如果用户需要更短的输出，工程师应该考虑调整提示词。限制大预言模型的输出长度对于类似ReAct的模型来说尤其重要，因为它们会持续的输出用户不需要的无用token。

## Sampling controls
大预言模型通常不会输出单个token，相反的，它们会在自己的词表中选择一个有可能的token。这些token会被采样，模型会根据这些token预测下一个token。temperature, top-k, top-p 三个参数可以控制模型的采样策略。

### Temperature

温度是控制token选择的随机程度。低的温数值会导致更加确定性的token选择，高的温度数值会导致更随机的token选择。0温度的时候会导致token选择变成贪心算法，总是选择概率最高的token。当温度变得越来越高是，token选择也越来越公平。

### Top-K and Top-P

Top-K和Top-P策略是两种token采样策略，是从最高可能性的一批token中筛选出适合预测下个token的策略。

* **Top-K** 策略是将token的可能性从高到低排序后，从高往低取K的策略。K值越大，模型的输出越是多样化和创造性；K值越小，模型的输出更加规律和符合事实。当K为1是，相当贪心算法
* **Top-P**策略是将token的可能性从高到低排序后，从高到低选择总和可能性为P的token。当P等于0时，这个策略相当贪心算法；当P等于1时，相当于所有token都有相同的可能性被选中。

### 混合策略

Temperature，Top-K，Top-P三种策略可以混合起来使用（todo）



## 提示词技巧

LLM在大量的数据上训练后，得到遵循指令的能力，因此能够理解prompt和根据prompt生成答案。LLM的这一点并不是完美的，用户提供的prompt越清晰，LLM就能产生越好的预测。（其实时预训练带给了LLM预测下一个字符的能力，SFT提供了如何遵循prompt的能力）。在LLM训练的过程中和预测过程中的产生的prompt技巧能够帮助用户产生和prompt更加相关的结果。

### 普通的prompt / zero shot

一个zero shot的prompt是prompt中最简单的。用户只需要提供任务的描述和一些内容，LLM就能根据这些内容开始工作。这些描述和内容可以是任何东西，例如：一个问题、一个故事的开头或者一些指令。zero shot意味着没有例子。

### One-shot & few-shot

提供例子是对编写prompt有用的方式。这些例子可以帮助LLM理解用户的问题。例子可以帮助用户引导LLM输出特定的结构和模式。One-shot prompt是提供一个列子；Few-shot prompt是提供多个例子，这与One-prompt相似，多个例子可以增加LLM遵循例子的概率。

例子的多寡取决于多个因素，包括任务的复杂程度，例子的质量，LLM的能力。多个例子受到模型输入长度的限制，一般来说，Few-shot需要三到五个例子。

## System, contextual and role prompting

系统，上下文和角色的prompt是引导模型预测的技术。他们分别聚焦在不同的层面

* System prompting 是用于设定LLM整体背景和目的，定义了模型所作内容的蓝图，例如：翻译、对评论进行评分等
* Contextual prompting提供了当前任务的具体的细节或者背景。这可以帮助模型理解所问问题的细节区别，并且做出相应的调整。
* Role prompting会给LLM分配一个特殊的角色和身份。这可以帮助LLM调用符合这个角色或者身份的知识库回答问题。

接下来，我们要思考的是这几种prompt会有相互覆盖的部分。例如一个指派LLM角色的prompt有可能有context。但是每一种prompt都为不同的目的服务。

* System prompt：定义模型基本功能和总体目的
* Contextual prompt：提供及时的、临时的和当前任务特定的信息，用于知道模型产生预测
* Role prompt：框定模型的输出风格，增加特异性和个人性

正确区分这三种方式，能够帮助用户理解prompt的框架，允许用户将这三种方式灵活的组合起来，指导模型输出，同时也能更好分析每种prompt是如何影响模型的输出。

（todo 补充例子）



## Step-back prompting（后退提示）

Step-back prompting是一种简单的提示技术，使得LLM能够进行抽象，推导出高级概念和基本原理，从而得到正确的答案。Step-back prompting会使得LLM思考一个特定问题的相关的一般问题，并且将这个一般问题的答案作为后继的特殊问题的答案。这项技术可以让LLM在解决特殊问题之前激活相关的背景知识和产生分解步骤。

通过更加广泛和更深层次的原则性思考，LLM可以产生更加准确和富有洞察力的答案。Step-back prompting鼓励LLM思考地更加判别性，从而能将它地知识更加创新使用起来。对比于直接提示，Step-back prompting能更好利用LLM自身参数中地知识，从而完成任务。

通过关注一般性地原则，而不是具体地细节，Step-back prompting可以减轻LLM回应中地偏见。





## Chain of Thought

Chain of Thought prompting是一个通过产生中间推理步骤提高LLM推理能力地技术。这又助于模型产生更加准确地答案。Chain of Thought和Zero-shot prompting结合起来回答一个推理问题的答案是是有挑战性的。用户可以结合Chain of Thought和Few-shot prompting一起使用，帮助LLM解决更加复杂地推理问题。

Chain of Thought拥有很多的优点。首先，它省时省力，而且非常有效，并且和现成的LLM兼容良好（因此无需进行微调）。Chain of Thought还可以提供可解释性，用户可以在LLM的回答中学习，并且看到LLM的推理步骤。如果中间过程出了差错，用户可以从中识别出来。Chain of Thought还具有鲁棒性，对比于其他prompting技术在不同LLM上表现有较大的差异，在不同的LLM上有着相似的效果。

Chain of Thought有着更长的输出，这是它的缺点，意味着相同的问题需要花费更长的时间和更多的钱。



## Self-Consistency

LLM在多个NLP任务中表现出令人惊讶的成绩，但是他的推理能力却无法因为模型的增大获得突破。就像上一节我们学习的Chain of Thought，LLM可以通过prompt产生与人类相似的推理步骤。然而，思维链使用了一种类似贪心解码的策略，制约了他的有效性。Self-Consistency prompting结合多次采样和大多数投票的方式，去产生不同推理路径和选择最一致性的回答。这种方式提高了LLM回答的准确率和连贯性。

Self-Consistency给出了准确答案的伪概率可能性，但是显然它的代价很高

Self-Consistency有以下几个步骤

1. 产生不同的推理路径：LLM在相同的prompt情况下推理数次。LLM被设置了一个很高的temperature，被鼓励生成不同的推理路径和对问题的不同的角度。
2. 在产生的回答里提取答案
3. 选择最一致性和最普遍的答案



## Tree of Thoughts(ToT)

现在我们非常熟悉思维链和一致性的prompting，我们可以开始查看思维树。与思维链直将问题按照一条推理步骤拆分开来不同，思维树允许LLM同时探索多条推理路径。这提高了LLM对需要探索答案的复杂问题的对应性。他的工作方式是维护一个思维树，每个思维代表一个连贯的思考路径，模型可以在不同的节点分支出来，探索不同的推理路径。



## ReAct(reason & act)

Reason and Act prompting允许LLM能够结合自然语言推理和外部工具（搜索，代码解释器等）来解决复杂任务，从而使得LLM能够执行某些操作。例如与外部API交互以检索信息，这是成为Agent的第一步。

Reason and Act模拟人类在真实环境中的为了解决问题所采取的行动。这个prompting在多个领域都比其他prompting方式来的优秀。

Reason and Act prompting是一个思考-行动的循环。LLM针对问题的第一次思考产生行动的计划。然后LLM通过按计划采取这些行动并观察这些行动的结果。LLM会通过这些观察更新它的推理和产生下一轮的行动的计划。这个思考-行动的循环会一直持续到问题被解决。



## Automatic Prompt Engineering

现在我们又拥有多种prompting的方法，可能你会因此感觉到困惑，不知道应该合理地使用那种。如果能将产生prompting的过程自动化就最好了。这里有个完成这个自动化过程的方法，不仅仅减轻了对人工输入的需求，还能提升模型在各项任务中的表现。

我们可以促进模型生成更多的提示，并且评估他们，有时修改中间较好的部分。重复上诉步骤。



## Code Prompting



### Prompts for writing code



### Prompts for explaining code



### Prompts for translating code



### Prompts for debugging and reviewing code



### What about multimodal prompting?



## Best Practices



