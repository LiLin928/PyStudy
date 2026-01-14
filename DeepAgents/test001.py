import os
from deepagents import create_deep_agent
from tavily import TavilyClient

# 1. 定义工具
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))  # 补充环境变量获取方式，更健壮

def internet_search(query: str):
    """运行网络搜索
    
    Args:
        query: 搜索关键词
        
    Returns:
        搜索结果（Tavily API 返回的结构化数据）
    """
    return tavily_client.search(query)

# 2. 定义系统提示词
# 注意：不需要手动添加关于 TodoList 或文件系统的指令，中间件会自动注入
system_prompt = """
你是一个专家级的研究助理。你的任务是进行彻底的调研并撰写报告。
请充分利用你的规划能力和文件系统来管理你的工作。
"""

# 3. 创建代理
# create_deep_agent 会自动装配 TodoListMiddleware, FilesystemMiddleware 等
agent = create_deep_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[internet_search],
    system_prompt=system_prompt,
)

# 4. 运行代理
# 代理现在是一个编译好的 LangGraph StateGraph
result = agent.invoke({
    "messages": [{"role": "user", "content": "详细调研 LangChain 的最新发展并写一份总结。"}]
})

# 可选：打印结果（方便查看执行输出）
print(result["messages"][-1]["content"])