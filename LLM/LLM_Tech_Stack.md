

# LLM inference technology stack

* # RL frameworks & algorithms for LLMS

  * ## RL frameworks

    * frameworks

    | 框架名称      | 主要维护方/来源       | 核心特点/设计侧重                                            | 代表性算法支持                 | 训练后端支持                         | 推理后端支持       |
    | ------------- | --------------------- | ------------------------------------------------------------ | ------------------------------ | ------------------------------------ | ------------------ |
    | **Verl**      | 字节Seed              | 功能全面，算法支持广泛，支持多轮训练和工具调用，追求高性能和可扩展性 | PPO, GRPO, DAPO, ReMax 等      | FSDP/FSDP2, Megatron-LM              | vLLM, SGLang       |
    | **OpenRLHF**  | OpenLLMAI, 字节，网易 | 早期流行开源框架，易用高性能，支持异步训练和Agentic RL       | PPO, GRPO, DPO, REINFORCE++ 等 | DeepSpeed ZeRO-3                     | vLLM               |
    | **TRL**       | Hugging Face          | 与Hugging Face生态紧密集成，API简洁，非常适合快速原型验证和中小规模实验。 | SFT, PPO, DPO, GRPO 等         | Hugging Face Trainer (支持DDP, FSDP) | Hugging Face, vLLM |
    | **AReaL**     | 蚂蚁集团              | **完全异步架构**的核心创新，旨在实现极致的训练吞吐量和可扩展性。 | GRPO, PPO 等                   | PyTorch FSDP, Megatron               | SGLang, vLLM       |
    | **ROLL**      | 阿里巴巴              | 基于Ray的多角色分布式设计，高度可配置，面向大规模生产和复杂智能体任务。 | GRPO, PPO, REINFORCE++ 等      | DeepSpeed, Megatron-LM               | vLLM, SGLang       |
    | **Slime**     | 智谱AI&清华           | 轻量级，**专注于无缝连接Megatron-LM训练框架和SGLang推理引擎**，追求极简与高性能。 | -                              | Megatron-LM (默认), FSDP             | SGLang             |
    | **NeMo-RL**   | NVIDIA                | NVIDIA官方出品，与硬件软件栈深度集成，强调生产级可扩展性和稳定性。 | SFT, DPO, PPO, GRPO 等         | Megatron-Core, FSDP2                 | TensorRT-LLM, vLLM |
    | **Cosmos-RL** | NVIDIA                | 采用**纯异步异构部署架构**，训练效率和容错能力突出，资源利用率高。 | -                              | -                                    | -                  |

    * 框架选择

  * ## RL algorithms

    * ### 综述

    * ### PPO

    * ### DPO

    * ### GRPO

    * 

* # Tool calling, MCP, Harmony format, OpenAI/Anthropic API

  * ## Tool calling

    Tool calling相对于function call是一个更加高级的概念，泛指一切AI Agent使用任何外部工具的能力。Function call是实现Tool calling的一种主流方式。

    样例

    ```python
    from openai import OpenAI
    import json
    import os
    import random
    
    # 初始化客户端，这里以阿里云百炼为例，其他平台类似
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 请替换为你的API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 1. 定义工具：告诉模型它可以使用的函数
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气情况",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市或县区，例如：北京市、杭州市",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"],  # location是必填参数
                },
            },
        }
    ]
    
    # 2. 实现工具函数：这是实际执行逻辑的代码
    def get_current_weather(location: str, unit: str = "celsius"):
        """模拟获取天气的函数，真实场景会调用天气API"""
        weather_data = {"北京": {"temp": 22, "condition": "晴"}}
        # 模拟数据获取，默认返回北京天气
        city_data = weather_data.get(location, weather_data["北京"])
        temp = city_data["temp"]
        return f"{location}的天气为{city_data['condition']}，温度{temp}度({unit})"
    
    # 3. 主程序逻辑
    def main():
        # 用户问题
        user_question = "请问北京和上海的天气怎么样？"
        messages = [{"role": "user", "content": user_question}]
        
        # 第一次调用模型：模型判断是否需要调用工具以及如何调用
        response = client.chat.completions.create(
            model="qwen-plus",  # 使用支持tool calling的模型
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自动决定是否调用工具
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # 检查模型是否要求调用工具
        if assistant_message.tool_calls:
            print("模型决定调用工具...")
            # 处理每个工具调用（模型可能同时调用多个工具）
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"调用函数: {func_name}, 参数: {args}")
                
                # 根据函数名执行对应的工具函数
                if func_name == "get_current_weather":
                    tool_result = get_current_weather(**args)
                else:
                    tool_result = f"错误: 未知函数 {func_name}"
                
                # 将工具执行结果添加到对话历史中
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # 必须匹配对应的tool_call id
                    "content": tool_result
                })
            
            # 第二次调用模型：让模型基于工具结果生成最终回答
            second_response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
            )
            
            final_reply = second_response.choices[0].message.content
            messages.append(second_response.choices[0].message)
            
            print(f"最终回复: {final_reply}")
        else:
            # 如果模型认为不需要调用工具，直接使用其回复
            print(f"直接回复: {assistant_message.content}")
    ```

    

  * ## MCP

    MCP是一个开放性的标准协议，目的是标准化AI大模型和外部数据的工具之间的交互方式。它的核心是提供一种统一的语言，是的任何支持MCP的AI应用（Client)都能无缝使用任何同样支持MCP的外部服务（Server)提供的工具和能力，而无需为每个组合单独开发适配代码。这种协议将应用和工具之间的复杂的点对点集成（MxN）简化成更线性的（M+N），提高了开发效率。Server和Clinet有两种对接方式Http和stdio

    Server

    ```python
    # mcp_server.py
    from mcp import MCPServer, Context  # 导入MCP核心库
    import logging
    
    # 创建MCP服务器实例
    server = MCPServer("calculator-tools")
    
    # 使用装饰器注册工具，使其可通过MCP协议调用
    @server.tool()
    def add(a: float, b: float) -> float:
        """一个加法计算工具。返回参数a和b的和。"""
        return a + b
    
    @server.tool()
    def divide(a: float, b: float) -> float:
        """一个除法计算工具。返回a除以b的结果。"""
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b
    
    if __name__ == "__main__":
        # 启动服务器，使用stdio方式通信，便于被客户端调用
        logging.basicConfig(level=logging.INFO)
        server.run(transport="stdio")
    ```

    Client

    ```python
    # mcp_client.py
    import asyncio
    from mcp import ClientSession, StdioServerParameters  # 导入MCP客户端相关库
    from mcp.client.stdio import stdio_client  # 导入stdio客户端实现
    
    async def main():
        # 1. 配置服务器启动参数：告诉客户端如何启动我们刚写的服务器
        server_params = StdioServerParameters(
            command="python",  # 执行命令
            args=["/path/to/your/mcp_server.py"]  # 服务器脚本路径
        )
    
        # 2. 建立与服务器的连接
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 3. 初始化会话，与服务器握手
                init_result = await session.initialize()
                print(f"已连接到MCP服务器: {init_result}")
    
                # 4. 向服务器请求可用的工具列表
                tools = await session.list_tools()
                print("可用的工具:", [tool.name for tool in tools])
    
                # 5. 模拟用户请求：计算12加5，再除以2
                user_request = "请先计算12加5，再用结果除以2。"
    
                # 6. （此处简化）在实际应用中，这里需要一个大模型来理解用户请求，
                # 并决定调用哪些工具及其参数。本例直接手动模拟这一决策。
                # 模型可能会决定先调用add(12, 5)，然后用结果调用divide(result, 2)
    
                # 模拟第一步：调用加法工具
                add_result = await session.call_tool("add", {"a": 12, "b": 5})
                first_step_result = add_result.content  # 假设返回17
                print(f"12 + 5 = {first_step_result}")
    
                # 模拟第二步：调用除法工具
                divide_result = await session.call_tool("divide", {"a": first_step_result, "b": 2})
                final_result = divide_result.content
                print(f"17 / 2 = {final_result}")
    
                print(f"\n最终答案: {final_result}")
    
    if __name__ == "__main__":
        asyncio.run(main())
    ```

    

  * ## Harmony format

    **Harmony Format**（Harmony 响应格式）是OpenAI为其开源模型gpt-oss配套设计的一套对话格式规范，旨在为推理、函数调用和模型行为元数据提供更丰富的结构。它通过引入角色层级、输出通道和语法约束等机制，彻底改变了AI模型与外部工具交互的方式。适用于多步推理和工具调用的任务，高稳定性的自动化流程和透明化的推理过程。

    #### 核心设计理念

    Harmony Format的核心设计体现了AI交互的三个重要趋势：

    **透明化（Transparency）**：模型的思考过程不再是黑箱，开发者可以观测、记录甚至干预其推理路径。通过analysis通道，模型会展示其内部的思维链（Chain-of-Thought），让开发者能够理解模型是如何一步步得出结论的。

    **可靠性（Reliability）**：通过语法约束机制，让模型与外部世界的交互（如工具调用）变得前所未有的稳定。使用`<|constrain|>`特殊令牌可以确保模型输出的JSON格式必定正确，大大提升了工具调用的成功率。

    **控制力（Control）**：通过角色和通道的分层，开发者获得了对模型行为更精细的控制权。System、Developer、User、Assistant、Tool五个角色形成了明确的指令优先级，而analysis、commentary、final三个通道则将模型的输出行为细粒度化

    #### 主要特性

    1. 角色层级系统：harmony format引入了明确的角色层级：System > Developer > User > Assistant > Tool。System角色存放元数据（如模型身份、知识截止日期、当前日期），Developer角色则包含开发者指令、可用工具定义等，优先级高于User输入。
    2. 多通道输出：Harmony format将助手的回复分为三个不同的通道。
       * **analysis通道**：存放模型的内部思考过程（思维链），这些内容不应直接展示给最终用户，但对开发者理解模型行为至关重要。
       * **commentary通道**：用于触发工具调用或向用户宣告行动计划。
       * **final通道**：最终呈现给用户的答案，经过完整的安全和质量对齐。
    3. 语法约束采样：通过`<|constrain|>json`令牌，模型在生成JSON参数时会激活语法约束采样机制，确保输出的每个Token组合起来都是语法合法的JSON。这解决了传统Function Calling中JSON格式错误的问题，将工具调用的成功率从"大概率"提升到了"必定"。

  * ## OpenAI/Anthropic API





* # Structured output / constraint decoding

  **结构化输出（Structured Output）**和**约束解码（Constraint Decoding）**是让大语言模型生成符合预定义格式（如JSON、XML、SQL等）的关键技术，已成为AI应用的核心能力。2025年，该领域的技术生态已日趋成熟，从最初的Prompt引导发展到模型原生支持的硬性接口化能力。

  * **Prompt project (可靠性85%)**：通过精心设计的提示词进行软性引导，是最简单但可靠性有限的方法，但不是100%有效。

    * 清晰的指令：在prompt里，使用明确的动作动词来指定期望的操作，并详细定义输出格式和内容。例如输出包含name, age和hobbies的JSON对象。
    * 少样本学习：模型通过这些示例学习到模式的细微之处，并泛化到新的输入上。这本质上是一种在推理时对模型进行“行为示范”的方法，能够显著提高输出的准确性和一致性。

  * **验证与修复框架（Guardrails等)**：在生成后进行"事后"保障，通过重试机制（Reask）自动验证和修正输出，确保最终结果合规。

    * 清晰的定义结构：使用Pydantic模型或JsonSchema等工具，详细定义期望的输出结构、字段类型、以及所需的验证规则和纠正措施。  

      pydantic

      ```python
      from pydantic import BaseModel, Field
      from typing import List
      
      # 使用 Pydantic 定义期望的输出结构
      class Book(BaseModel):
          title: str = Field(description="书名")
          author: str = Field(description="作者")
          publication_year: int = Field(description="出版年份", ge=1000, le=2030) # 设置年份范围约束
          genres: List[str] = Field(description="书籍流派列表", min_items=1)
          summary: str = Field(description="书籍摘要", min_length=10)
      ```

      guardrails

      ```python
      from guardrails import Guard
      import openai
      
      # 1. 从 Pydantic 模型创建 Guard
      guard = Guard.from_pydantic(output_class=Book, prompt="请生成一本虚构书籍的信息。"
                                               )
      
      # 2. 使用 Guard 来包装你的 LLM 调用
      # Guardrails 会自动修改提示词（Prompt）并解析、验证 LLM 的输出
      raw_llm_response, validated_response, *other_info = guard(
          llm_api=openai.chat.completions.create, # 指定 LLM API
          model="gpt-3.5-turbo",                    # 指定模型
          max_tokens=512,
      )
      
      # 3. 查看结果
      print("✅ 原始 LLM 响应:\n", raw_llm_response)
      print("\n✅ 经过 Guardrails 验证和解析后的结构化数据:\n", validated_response)
      print("\n✅ 你可以像操作普通 Pydantic 对象一样访问字段:")
      print(f"   书名: {validated_response.title}")
      print(f"   作者: {validated_response.author}")
      ```

    * 自动验证和修复：模型生成输出后，守卫对象会根据预先定义的结构规范对输出进行自动验证。如果发现输出格式无效、字段缺失、或内容不符合要求，它将自动执行纠正措施。

      例如guardrails的关键机制：

      * **结构化输出**：`Guard.from_pydantic`方法的核心作用是引导 LLM 输出规范的 JSON 对象。Guardrails 会在后台自动优化提示词（例如，添加类似 “You must return a JSON object that follows this schema...” 的指令），并尝试将 LLM 的原始文本响应解析成 JSON，再根据 Pydantic 模型进行验证
      * **自动验证**：如果 LLM 的输出无法被解析为 JSON，或者解析后的数据不符合 Pydantic 模型定义的约束（例如，`publication_year`超出了指定范围），Guardrails 会自动处理这些错误。其默认策略是 **“重新询问”（Reask）**，即自动将验证失败的信息和错误反馈给 LLM，要求它重试
      * **返回值**：`guard()`调用主要返回两个值：
        - `raw_llm_response`: LLM 的原始回复文本。
        - `validated_response`: 一个已实例化的 `Book`对象（Pydantic 模型实例），你可以直接通过属性（如 `.title`）访问其字段，非常方便集成到后续逻辑中

  * **约束解码（Constrained Decoding）**：在模型生成过程中进行"事前"干预，通过有限状态机（FSM）或上下文无关文法（CFG）动态约束输出空间，实现100%格式准确度。
    * 2017-2022早期阶段：约束解码技术最早由Hokamp和Liu在2017年提出，随后Post和Vilar在2018年提出了动态波束分配（Dynamic Beam Allocation）的改进方法。这一阶段的核心思想是通过给定候选词汇列表，强制模型生成包含所有指定词汇的序列。Fairseq框架实现了Vectorized Lexically Constrained Decoding，用户可以通过`--constraints`参数指定约束词汇，模型在生成时确保输出包含这些词汇。
    * 2022-2023结构化输出兴起：随着LLM应用场景的扩展，单纯的词汇约束已无法满足复杂结构化输出的需求。这一时期出现了基于有限状态机（FSM）和确定性有限自动机（DFA）的约束解码方案。这些方法将JSON Schema等结构化模式转换为正则表达式，再编译为DFA，在解码过程中动态限制可生成的token集合。
    * 2024-2025通用文法约束：2024年，约束解码技术迎来重大突破。X-Grammar等先进方案采用扩展巴科斯范式（EBNF）来表达上下文无关文法（CFG），并通过下推自动机（PDA）数据结构实现复杂语法解析。这种方案能够处理SQL查询、图查询语言Cypher等具有递归和嵌套特性的复杂结构，表达能力显著增强。
    * 创新点：
      * **Sketch-Guided Constrained Decoding（SketchGCD）**：该方法将黑盒LLM的无约束输出视为"草图"，利用本地部署的辅助模型进行精炼和修正，在不访问黑盒模型内部机制的情况下实现了约束解码效果。
      * **DSCD（Large Language Model Detoxification with Self-Constrained Decoding）**：是一种无需参数微调的自约束解码方法。该方法通过早期退出机制，在解码过程中动态调整token分布，强化安全层、弱化毒性层和幻觉层的作用，实现了高效去毒的同时保持生成流畅性。
      * **COIECD（Contextual Information Entropy Constrained Decoding）**：是一种自适应解码方法，通过上下文信息熵约束来识别和解决知识冲突。该方法能够提高模型对冲突上下文的忠实度，同时在非冲突环境中保持高性能，在现实数据集中的知识冲突场景下表现出强大的性能和鲁棒性。
  * **监督式微调（Supervised Fine-Tuning, SFT）**：通过数据集训练，使模型内化结构化输出的规则。
  * **强化学习优化（Reinforcement Learning Optimization）**：采用奖励机制，突破SFT的性能瓶颈。
  * **接口化能力（API Capabilities）**：将复杂技术抽象为简单易用的API功能。一般大公司例如阿里上的模型API都通过接口化的约束解码或者验证与修复框架的功能完成对大模型输出的辅助。



* # High-performance kernels: Attention, GEMM, sampling, sorting

  ## GEMM（通用矩阵乘法）优化

  **GEMM**是许多计算任务的基石，尤其是在AI中，它可能占据Transformer模型70%以上的计算时间。其优化核心在于**最大化数据复用率和硬件计算单元利用率**。**GEMM的计算复杂性**体现在其O(M*N*K)的时间复杂度上，但更重要的是其**内在的并行性和数据复用特性**为硬件优化提供了巨大空间。

  * **循环分块与数据局部性**：通过将大矩阵划分为适合高速缓存（如L1、共享内存）的小块（Tiling），可以显著减少访问慢速全局内存的次数。例如使用Block Tile，Warp Tile和Thread Tile来优化。

    * GEMM的朴素定义

      ```python
      #输入：矩阵A(M行K列），矩阵B(K行N列)
      #输出：矩阵C(M行N列)
      
      for i in range(M-1):
          for j in range(N-1):
              C[i][j] = 0;
              for p in range(K-1):
                  C[i][j] += A[i][p] * B[p][j]
      ```

      [reference1 CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275)

      [reference2 Cuda矩阵乘法GeMM性能优化]([(26 封私信) 1. Cuda矩阵乘法GeMM性能优化 - 知乎](https://zhuanlan.zhihu.com/p/593462636))

      [reference3 一步步优化 GEMM by Tensorcore](https://zhuanlan.zhihu.com/p/638522893)

  * **双缓冲与异步数据搬运**：这项技术旨在重叠计算与数据搬运，以隐藏内存访问延迟。其原理是设置两块缓冲区：一块用于当前计算，另一块用于异步预加载下一个计算所需的数据。实测数据显示，这能将计算单元利用率从45%提升至75%以上。

  * **寄存器优化**：通过寄存器分块和循环展开，可以提高指令级并行度。例如，将4x4的小块数据保留在寄存器中进行累加，减少对共享内存的访问，此类优化能提升40%以上的寄存器利用率。

  * **利用专用的硬件单元**：在现代GPU（如NVIDIA的V100、A100、H100）上，可以使用**Tensor Cores**。这些是专门为执行小型矩阵乘法（如4x4x4）而设计的硬件单元，能在一个周期内完成大量操作，相比传统的CUDA核心，吞吐量有数量级的提升。要使用Tensor Core，通常需要采用特定的数据布局和精度（如FP16）。