# 智能客服技术手册阅读笔记

## LangGraph

**LangGraph**是LangChain团队开发的开源框架，专门用于构建基于大型语言模型(LLM)的有状态、多角色应用程序。它通过图结构(Graph)实现复杂的工作流编排，特别适合需要循环执行、条件分支和多智能体协作的场景

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 定义状态结构
class State(TypedDict):
    input: str
    output: str

# 创建图
workflow = StateGraph(State)

# 定义节点
def node_1(state: State) -> State:
    state["output"] = f"处理输入: {state['input']}"
    return state

def node_2(state: State) -> State:
    state["output"] += " -> 已完成"
    return state

# 添加节点和边
workflow.add_node("node_1", node_1)
workflow.add_node("node_2", node_2)
workflow.add_edge(START, "node_1")
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", END)

# 编译和执行
graph = workflow.compile()
result = graph.invoke({"input": "Hello, LangGraph!"})
print(result)  # {'input': 'Hello, LangGraph!', 'output': '处理输入: Hello, LangGraph! -> 已完成'}
```



使用node edge等概念进行图的创建，

## 路由作用

路由是用在将客户的提问进行分类，进入不同的业务模块。