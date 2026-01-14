# 核心依赖导入
import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from dotenv import load_dotenv

# 尝试导入 Rich 库用于美化输出（非必需）
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
    print("Rich 库已加载，将使用美化输出")
except ImportError:
    RICH_AVAILABLE = False
    print("ℹ Rich 库未安装，使用标准输出")

# 加载环境变量（override=True 确保覆盖系统环境变量）
load_dotenv(override=True)

# 读取配置
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL")
tavily_key = os.environ.get("TAVILY_API_KEY")

print(f"deepseek_api_key: {deepseek_api_key[:10]}...{deepseek_api_key[-10:]}")
print(f"deepseek_base_url: {deepseek_base_url}")
print(f"tavily_key: {tavily_key[:10]}...{tavily_key[-10:]}")

# 初始化 Tavily 客户端
tavily_client = TavilyClient(api_key=tavily_key)

print("Tavily 客户端初始化成功")
print("   功能：提供实时互联网搜索能力")
print("   配额：1000 次/月（免费版）")

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """
    运行网络搜索
    
    这是一个用于网络搜索的工具函数，封装了 Tavily 的搜索功能。
    
    参数说明：
    - query: 搜索查询字符串，例如 "Python 异步编程教程"
    - max_results: 最大返回结果数量，默认为 5
    - topic: 搜索主题类型，可选 "general"（通用）、"news"（新闻）或 "finance"（金融）
    - include_raw_content: 是否包含原始网页内容，默认为 False
    
    返回：
    - 搜索结果字典，包含标题、URL、摘要等信息
    """
    try:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return result
    except Exception as e:
        # 异常处理：返回错误信息而非抛出异常
        # 这样 LLM 可以理解出错并尝试其他策略
        return {"error": f"搜索失败: {str(e)}"}

print("搜索工具创建完成")

# 测试搜索功能
print("开始测试搜索工具...\n")

test_result = internet_search("帮我检索一下 DeepSeek-v3.2 最新模型的特性", max_results=3)

print("搜索测试结果：")
print(f"结果数量: {len(test_result.get('results', []))}")

# 显示第一条结果
if test_result.get('results'):
    first = test_result['results'][0]
    print(f"\n标题: {first.get('title', 'N/A')}")
    print(f"链接: {first.get('url', 'N/A')}")
    print(f"摘要: {first.get('content', 'N/A')[:150]}...")
    print("\n搜索工具测试通过！")
else:
    print("\n未获取到搜索结果，请检查网络连接和 API 配额")



# 加载 .env 文件中的环境变量
load_dotenv(override=True)

# 检查环境变量是否已设置
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL")


if not deepseek_api_key:
    print("警告：未找到 OPENAI_API_KEY 或 DEEPSEEK_API_KEY 环境变量")
    print("请至少设置其中一个：")
    print("  export OPENAI_API_KEY='your-openai-api-key'")
    print("  或")
    print("  export DEEPSEEK_API_KEY='your-deepseek-api-key'")
    raise ValueError("至少需要设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")

print("环境变量检查完成")
if deepseek_base_url:
    print("DEEPSEEK_BASE_URL 已设置")
    print(f"DEEPSEEK_BASE_URL: {deepseek_base_url}")
    
# 使用模型字符串（LangChain 会自动识别供应商）
model = init_chat_model(
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
    model_provider="deepseek",
    model="deepseek-chat"
)
# 测试模型
print("测试 DeepSeek 模型连接...\n")

test_response = model.invoke("请用一句话介绍你自己。")

print("DeepSeek-v3.2 的回复：")
print(test_response.content)
print("\n模型测试通过！DeepSeek 已就绪")

# 系统提示词：指导智能体成为专家研究员
research_instructions = """
您是一位资深的研究人员。您的工作是进行深入的研究，然后撰写一份精美的报告。

您可以通过互联网搜索引擎作为主要的信息收集工具。

## `互联网搜索`

使用此功能针对给定的查询进行互联网搜索。您可以指定要返回的最大结果数量、主题以及是否包含原始内容。

在进行研究时：
1. 首先将研究任务分解为清晰的步骤
2. 使用互联网搜索来收集全面的信息
3. 如果内容太大，将重要发现保存到文件中
4. 将信息整合成一份结构清晰的报告
5. 务必引用你的资料来源
"""

print("系统提示词已定义")

# 创建 Deep Agent
agent = create_deep_agent(
    model=model,                           # 使用 DeepSeek 模型
    tools=[internet_search],               # 传入搜索工具
    system_prompt=research_instructions    # 系统提示词
)

print("Deep Agent 创建成功！")

query = "请帮我介绍一下DeepSeek-v3.2 最新模型的特性，注意：请用中文回答！"

result = agent.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

print("Deep Agent 回复：")
print(result["messages"][-1]["content"])

#create_deep_agent 的内部机制解读
def print_agent_tools(agent):
    """
    打印 Agent 中加载的所有工具
    包括用户自定义工具、文件系统工具、系统工具等
    """
    # 获取 agent 的 nodes (LangGraph 的节点)
    if hasattr(agent, 'nodes') and 'tools' in agent.nodes:
        tools_node = agent.nodes['tools']

        # tools_node 是 PregelNode，真正的 ToolNode 在 bound 属性中
        if hasattr(tools_node, 'bound'):
            tool_node = tools_node.bound

            # 从 ToolNode 获取工具
            if hasattr(tool_node, 'tools_by_name'):
                tools = tool_node.tools_by_name

                # 分类工具
                user_tools = []
                filesystem_tools = []
                system_tools = []

                for tool_name, tool in tools.items():
                    tool_info = {
                        'name': tool_name,
                        'description': getattr(tool, 'description', '无描述')
                    }

                    # 分类
                    if tool_name in ['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'execute']:
                        filesystem_tools.append(tool_info)
                    elif tool_name in ['write_todos', 'task']:
                        system_tools.append(tool_info)
                    else:
                        user_tools.append(tool_info)

                # 打印加载工具的输出
                _print_tools_rich(user_tools, filesystem_tools, system_tools)
             
            else:
                print("无法获取工具列表 (tools_by_name 不存在)")
        else:
            print("无法获取工具列表 (bound 属性不存在)")
    else:
        print("无法获取工具列表 (nodes 结构不符合预期)")

def _print_tools_rich(user_tools, filesystem_tools, system_tools):
    """使用 Rich 库美化打印工具列表"""
    console.print()

    # 创建表格
    table = Table(title="Agent 加载的工具列表", show_header=True, header_style="bold magenta")
    table.add_column("类别", style="cyan", width=20)
    table.add_column("工具名称", style="green", width=20)
    table.add_column("描述", style="white", width=60)

    # 添加用户工具
    for i, tool in enumerate(user_tools):
        category = "用户工具" if i == 0 else ""
        desc = tool['description'][:80] + "..." if len(tool['description']) > 80 else tool['description']
        table.add_row(category, tool['name'], desc)

    # 添加文件系统工具
    for i, tool in enumerate(filesystem_tools):
        category = "文件系统工具" if i == 0 else ""
        desc = tool['description'][:80] + "..." if len(tool['description']) > 80 else tool['description']
        table.add_row(category, tool['name'], desc)

    # 添加系统工具
    for i, tool in enumerate(system_tools):
        category = "系统工具" if i == 0 else ""
        desc = tool['description'][:80] + "..." if len(tool['description']) > 80 else tool['description']
        table.add_row(category, tool['name'], desc)

    console.print(table)

    # 打印统计
    total = len(user_tools) + len(filesystem_tools) + len(system_tools)
    console.print(Panel(
        f"[bold green]共计 {total} 个工具[/bold green]\n\n"
        f"• 用户工具: {len(user_tools)} 个\n"
        f"• 文件系统工具: {len(filesystem_tools)} 个\n"
        f"• 系统工具: {len(system_tools)} 个",
        title="统计信息",
        border_style="green"
    ))
    console.print()