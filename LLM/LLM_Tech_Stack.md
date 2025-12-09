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

  * ## MCP

  * ## Harmony format

  * ## OpenAI/Anthropic API

  * 