# 核心依赖导入
import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from deepagents import create_deep_agent

# 加载环境变量
load_dotenv(override=True)

# 读取配置
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
tavily_key = os.environ.get("TAVILY_API_KEY")

print("环境变量加载完成")

from langchain.chat_models import init_chat_model

# 使用模型字符串（LangChain 会自动识别供应商）
model = init_chat_model(
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
    model_provider="deepseek",
    model="deepseek-chat"
)

# 初始化 Tavily 客户端
tavily_client = TavilyClient(api_key=tavily_key)

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

# 文件写入工具
def write_local_file(file_path: str, content: str) -> dict:
    """
    将内容写入本地文件
    
    这是一个用于将内容保存到本地文件的工具函数。
    
    参数说明：
    - file_path: 文件路径，例如 "report.md" 或 "./reports/research_report.md"
    - content: 要写入文件的内容（字符串）
    
    返回：
    - 包含操作结果的字典，如果成功则返回 {"status": "success", "file_path": file_path}
    - 如果失败则返回 {"status": "error", "error": "错误信息"}
    """
    try:
        # 确保目录存在
        import os
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "file_path": file_path,
            "message": f"文件已成功保存到: {file_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"写入文件失败: {str(e)}"
        }

print("文件写入工具创建完成")

# 系统提示词
research_instructions = """您是一位资深的研究人员。您的工作是进行深入的研究，然后撰写一份精美的报告。

您可以通过互联网搜索引擎作为主要的信息收集工具。

## 可用工具

### `互联网搜索`
使用此功能针对给定的查询进行互联网搜索。您可以指定要返回的最大结果数量、主题以及是否包含原始内容。

### `写入本地文件`
使用此功能将研究报告保存到本地文件。当您完成研究并生成报告后，请使用此工具将完整的报告内容保存到文件中。
- 文件路径建议使用 .md 格式（Markdown），例如 "research_report.md" 或 "./reports/报告名称.md"
- 请确保报告内容完整、结构清晰，包含所有章节和引用来源

## 工作流程

在进行研究时：
1. 首先将研究任务分解为清晰的步骤
2. 使用互联网搜索来收集全面的信息
3. 将信息整合成一份结构清晰的报告
4. **重要**：完成报告后，务必使用 `写入本地文件` 工具将完整报告保存到本地文件
5. 务必引用你的资料来源

**注意**：请确保在完成研究后，将完整的报告内容保存到文件中，这样用户可以方便地查看和保存报告。
"""

# 创建 Deep Agent
agent = create_deep_agent(
    model=model,
    tools=[internet_search, write_local_file],
    system_prompt=research_instructions
)

print("DeepAgents 创建成功！")

# 导入必要的库
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.json import JSON

# 导入 Rich 库
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.json import JSON
    RICH_AVAILABLE = True
    console = Console()
    print("Rich 库已加载，将使用美化输出")
except ImportError:
    RICH_AVAILABLE = False
    console = None
    print("Rich 库未安装，将使用标准输出")
    
    
def debug_agent(query: str, save_to_file: str = None):
    """
    运行智能体并打印中间过程（使用 Rich 美化输出）
    
    参数:
        query: 用户查询
        save_to_file: 保存最终输出到文件(可选)
    
    返回:
        str: 最终的研究报告
    """

    console.print(Panel.fit(
        f"[bold cyan]查询:[/bold cyan] {query}",
        border_style="cyan"
    ))
    
    step_num = 0
    final_response = None
    
    # 实时流式输出
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values"
    ):
        step_num += 1
        
        console.print(f"\n[bold yellow]{'─' * 80}[/bold yellow]")
        console.print(f"[bold yellow]步骤 {step_num}[/bold yellow]")
        console.print(f"[bold yellow]{'─' * 80}[/bold yellow]")
        
        if "messages" in event:
            messages = event["messages"]
            
            if messages:
                msg = messages[-1]
                
                # 保存最终响应
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                    final_response = msg.content
                
                # AI 思考
                if hasattr(msg, 'content') and msg.content:
                    # 如果内容太长,只显示前300字符作为预览
                    content = msg.content
                    if len(content) > 300 and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        preview = content[:300] + "..."
                        console.print(Panel(
                            f"{preview}\n\n[dim](内容较长,完整内容将在最后显示)[/dim]",
                            title="[bold green]AI 思考[/bold green]",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel(
                            content,
                            title="[bold green]AI 思考[/bold green]",
                            border_style="green"
                        ))
                
                # 工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_info = {
                            "工具名称": tool_call.get('name', 'unknown'),
                            "参数": tool_call.get('args', {})
                        }
                        console.print(Panel(
                            JSON(json.dumps(tool_info, ensure_ascii=False)),
                            title="[bold blue]工具调用[/bold blue]",
                            border_style="blue"
                        ))
                
                # 工具响应
                if hasattr(msg, 'name') and msg.name:
                    response = str(msg.content)[:500]
                    if len(str(msg.content)) > 500:
                        response += f"\n... (共 {len(str(msg.content))} 字符)"
                    
                    console.print(Panel(
                        response,
                        title=f"[bold magenta]工具响应: {msg.name}[/bold magenta]",
                        border_style="magenta"
                    ))
    
    console.print("\n[bold green]任务完成![/bold green]\n")
    
    return final_response

print("调试函数已创建")

# 示例：使用调试函数运行研究任务
query = "详细调研 LangChain DeepAgents 框架的核心特性，并写一份结构化的总结报告。"

# 使用调试函数）
result = debug_agent(query)