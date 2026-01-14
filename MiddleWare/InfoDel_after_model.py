from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat")

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """模型调用后，删除最早的两条消息"""
    messages = state["messages"]

    if len(messages) > 4:
        removed = [m.id for m in messages[:2]]
        print(f"🧹 删除前两条消息，ID: {removed}")
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    model,
    tools=[],
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)