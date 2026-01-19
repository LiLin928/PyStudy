# 基础库导入
import os
import sys
from pathlib import Path
from typing import List, Callable, Any, Optional
from typing_extensions import TypedDict

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(override=True)

# LangChain 1.1 核心导入
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

print("核心库导入成功")

# 添加项目路径（用于导入自定义模型）
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入自定义的 DeepSeek 模型适配器
from skill_system.models import DeepSeekReasonerChatModel

# 检查 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("未设置 DEEPSEEK_API_KEY")
    print("   请在 .env 文件中添加: DEEPSEEK_API_KEY=your-key")
else:
    print(f"API Key 已配置 (前8位: {api_key[:8]}...)")

# 创建模型实例
model = DeepSeekReasonerChatModel(
    api_key=api_key,
    model_name="deepseek-reasoner",
    temperature=0.7
)

print(f"DeepSeek 模型已创建")

# # 使用 MessagesState 而不是 TypedDict
from langgraph.graph import MessagesState
from typing import Annotated, List

# 第一种模式：替换模式
# # 定义 reducer 函数
# def skill_list_reducer(current: List[str], new: List[str]) -> List[str]:
#     """替换模式：用新列表替换旧列表"""
#     return new

# # 使用 MessagesState 作为基类
# class SkillState(MessagesState):
#     """
#     Skill 状态 Schema
    
#     使用 MessagesState 作为基类，它已经包含了 messages 字段
#     我们只需要添加 skills_loaded 字段
#     """
#     skills_loaded: Annotated[List[str], skill_list_reducer] = []


# 第二种模式：累计模式
# 修改 reducer 函数为累积模式
def skill_list_accumulator(current: List[str], new: List[str]) -> List[str]:
    """
    累积模式：合并已加载的 Skills
    保持所有已加载的技能，而不是替换
    """
    if not current:
        return new
    # 合并并去重，保持顺序
    combined = current + [s for s in new if s not in current]
    return combined

# 使用累积模式的 reducer
class SkillState(MessagesState):
    """
    Skill 状态 Schema
    """
    skills_loaded: Annotated[List[str], skill_list_accumulator] = []  # 改为累积模式
# ==================== Loader 工具 ====================
# 这些工具始终可见，用于加载其他技能

from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool
def skill_data_analysis(runtime) -> Command:
    """
    加载数据分析技能。
    """
    instructions = """数据分析技能已成功加载！
    
现在你可以使用以下工具：
• calculate_statistics(numbers): 计算一组数字的统计信息
• generate_chart(data, chart_type): 生成数据图表

请继续使用这些工具完成用户的数据分析任务。"""
    
    return Command(
        update={
            "messages": [ToolMessage(
                content=instructions,
                tool_call_id=runtime.tool_call_id
            )],
            "skills_loaded": ["data_analysis"]  # 关键：直接更新状态
        }
    )


@tool
def skill_text_processing(runtime) -> Command:
    """
    加载文本处理技能。
    
    调用此工具后，你将获得以下文本处理相关的工具：
    - summarize_text: 生成文本摘要
    - extract_keywords: 提取关键词
    
    使用场景：当用户需要处理文本、生成摘要或提取关键信息时，
    请先调用此工具加载文本处理技能。
    """
    instructions = """文本处理技能已成功加载！
    
现在你可以使用以下工具：
• summarize_text(text, max_length): 生成文本摘要
• extract_keywords(text, num_keywords): 提取关键词

请继续使用这些工具完成用户的文本处理任务。"""
    
    return Command(
        update={
            "messages": [ToolMessage(
                content=instructions,
                tool_call_id=runtime.tool_call_id
            )],
            "skills_loaded": ["text_processing"]  # 关键：直接更新状态
        }
    )


# ==================== 数据分析工具 ====================
# 这些工具只有在加载了 data_analysis 技能后才可见
@tool
def calculate_statistics(numbers: List[float]) -> str:
    """
    计算一组数字的统计信息，包括平均值、最大值、最小值、标准差等。
    
    Args:
        numbers: 要分析的数字列表
    """
    import statistics
    
    if not numbers:
        return "错误: 数字列表为空"
    
    result = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
    
    if len(numbers) > 1:
        result["stdev"] = statistics.stdev(numbers)
    
    return f"统计结果: {result}"


@tool
def generate_chart(data: List[float], chart_type: str = "bar") -> str:
    """
    根据数据生成图表（模拟）。
    
    Args:
        data: 数据列表
        chart_type: 图表类型 (bar, line, pie)
    """
    return f"已生成 {chart_type} 图表，包含 {len(data)} 个数据点"


# ==================== 文本处理工具 ====================
# 这些工具只有在加载了 text_processing 技能后才可见

@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    生成文本摘要。
    
    Args:
        text: 要摘要的文本
        max_length: 摘要最大长度
    """
    if len(text) <= max_length:
        return f"摘要: {text}"
    return f"摘要: {text[:max_length]}..."


@tool
def extract_keywords(text: str, num_keywords: int = 5) -> str:
    """
    从文本中提取关键词。
    
    Args:
        text: 要分析的文本
        num_keywords: 要提取的关键词数量
    """
    # 简单模拟：取前几个单词
    words = text.split()[:num_keywords]
    return f"关键词: {', '.join(words)}"


# 组织工具
LOADER_TOOLS = [skill_data_analysis, skill_text_processing]
DATA_ANALYSIS_TOOLS = [calculate_statistics, generate_chart]
TEXT_PROCESSING_TOOLS = [summarize_text, extract_keywords]
ALL_TOOLS = LOADER_TOOLS + DATA_ANALYSIS_TOOLS + TEXT_PROCESSING_TOOLS

print("工具定义完成")
print(f"   Loader 工具 ({len(LOADER_TOOLS)}): {[t.name for t in LOADER_TOOLS]}")
print(f"   数据分析工具 ({len(DATA_ANALYSIS_TOOLS)}): {[t.name for t in DATA_ANALYSIS_TOOLS]}")
print(f"   文本处理工具 ({len(TEXT_PROCESSING_TOOLS)}): {[t.name for t in TEXT_PROCESSING_TOOLS]}")
print(f"   总计: {len(ALL_TOOLS)} 个工具")

# 技能到工具的映射
SKILL_TOOL_MAPPING = {
    "data_analysis": DATA_ANALYSIS_TOOLS,
    "text_processing": TEXT_PROCESSING_TOOLS,
}

def get_tools_for_skills(skills_loaded: List[str]) -> List[BaseTool]:
    """
    根据已加载的技能列表，返回应该暴露给模型的工具
    
    核心逻辑：
    1. Loader 工具始终包含
    2. 根据 skills_loaded 添加对应的技能工具
    
    Args:
        skills_loaded: 已加载的技能名称列表
        
    Returns:
        过滤后的工具列表
    """
    # 始终包含 Loader 工具
    tools = list(LOADER_TOOLS)
    
    # 根据已加载的技能添加对应工具
    for skill_name in skills_loaded:
        if skill_name in SKILL_TOOL_MAPPING:
            tools.extend(SKILL_TOOL_MAPPING[skill_name])
    
    return tools


# 测试工具过滤函数
print("测试 get_tools_for_skills 函数:")
print(f"\n1. skills_loaded = []")
tools = get_tools_for_skills([])
print(f"   返回 {len(tools)} 个工具: {[t.name for t in tools]}")

print(f"\n2. skills_loaded = ['data_analysis']")
tools = get_tools_for_skills(['data_analysis'])
print(f"   返回 {len(tools)} 个工具: {[t.name for t in tools]}")

print(f"\n3. skills_loaded = ['data_analysis', 'text_processing']")
tools = get_tools_for_skills(['data_analysis', 'text_processing'])
print(f"   返回 {len(tools)} 个工具: {[t.name for t in tools]}")