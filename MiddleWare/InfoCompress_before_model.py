from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from typing import Any

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat")

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """在模型调用前修剪消息历史，只保留前1+后3条。"""
    messages = state["messages"]

    # 只有超过 4 条消息才裁剪
    if len(messages) <= 4:
        return None

    # 保留首条 System 提示 + 最近3条
    first_msg = messages[0]
    new_messages = [first_msg] + messages[-3:]

    print(f"✂️ 修剪消息：从 {len(messages)} 条 → {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)