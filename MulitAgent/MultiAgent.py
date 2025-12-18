import asyncio
from operator import add
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AnyMessage,HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.config import get_stream_writer
from langgraph.types import Checkpointer 
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
#from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek


load_dotenv()

modelUrl=(os.environ["ModelUrl"])
modelKey=(os.environ["ModelKey"])
modelName=(os.environ["ModelName"])
mapMcpKey=(os.environ["MapMcpKey"])
DEEPSEEK_API_KEY =modelKey

DASHSCOPE_API_KEY=(os.environ["DASHSCOPE_API_KEY"])
DashScopeEmbeddingModel=(os.environ["DashScopeEmbeddingModel"])

nodes=["supervisor","travel","joke","couplet","other"]

llm=ChatOpenAI(
    base_url=modelUrl,
    model=modelName,
    api_key=modelKey
)

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add]
    type:str
def supervisor_node(state:State):
    writer=get_stream_writer()
    # writer("node",">>> supervisor_node")
     # 根据用户的问题，对问题进行分类，分类结果存到type当中
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
如果用户的问题是和旅游线路规划相关的，那就返回 travel 。
如果用户的问题是希望讲一个笑话，那就返回 joke 。
如果用户的问题是希望对一对联，那就返回 couplet 。
如果是其他的问题，返回 other 。
除了这几个选项外，不要返回任何其他的内容。"""
    message_text =get_message_content(state["messages"][0])
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message_text }
    ]
    if "type" in state:     
        writer({"supervisor_node":f"已经获得问题分类结果：{state['type']}"})   
        return {"type":"END"}
    else:
        response=llm.invoke(prompts)
        typeRes=response.content
        writer({"supervisor_node":f"问题分类结果：{typeRes}"})
        if typeRes in nodes:
            return {"type":typeRes}
        else:    
            return ValueError("type is not in (travel,joke,couplet,other)")
async def travel_node(state:State):
    # print(">>> travel_node")
    writer=get_stream_writer()
    # writer("node",">>> travel_node")
    
    # 高德地图MCP
    client=MultiServerMCPClient(
        {
            # "amap-maps": {
            #     "url": "https://mcp.amap.com/mcp?key="+mapMcpKey,
            #     "transport": "streamable_http"
            # },
            "amap-maps": {
                "command": "npx",
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                "env": {
                    "AMAP_MAPS_API_KEY": mapMcpKey
                },
                "transport": "stdio"
            }
        }
    )
    
    prompt = "你是一个专业的履行规划大师，跟据用户的问题，生成一个旅游路线规划。请用中文回答，并返回不超过100字的结果"
    message_text =get_message_content(state["messages"][0])
    prompts = [
        {"role": "user", "content":message_text}
    ]
    #tools=asyncio.run(client.get_tools())
    tools=await client.get_tools()
    model = ChatDeepSeek(model="deepseek-chat",
    api_key=modelKey)

    # 3.创建Agent
    agent1 = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt
    )

    # 4.运行Agent获得结果
    response =await agent1.ainvoke(
        {"messages": prompts}
    )
    # openai的方法报错
    # agent=create_agent(model=llm,tools=tools,system_prompt=prompt)
    # response=await agent.ainvoke({"messages":[{"role": "user", "content": message_text}]})
    writer({"travel_node":f"旅游路线规划结果：{response["messages"][-1].content}"})
    return {"messages":[HumanMessage(content=response["messages"][-1].content)],"type":"travel"}
def joke_node(state:State):
    writer=get_stream_writer()
    # writer("node",">>> joke_node")
    prompt = "你是一个笑话大师，跟据用户的问题，写一个不超过100个字的笑话。"
    message_text=get_message_content(state["messages"][0])
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message_text}
    ]
    response=llm.invoke(prompts)
    jokeRes=response.content
    writer({"joke_node":f"笑话结果：{jokeRes}"})
    return {"messages":[HumanMessage(content=jokeRes)],"type":"joke"}
def couplet_node(state:State):
    # print(">>> couplet_node")
    writer=get_stream_writer()
    # writer("node",">>> couplet_node")
    prompt_template=ChatPromptTemplate.from_messages([
        ("system","""
         你是一个专业的对联大师，你的任务是跟据用户给出的上联，设计一个下联。
         回答时，可以参考下面的参考对联。
         参考对联：
            {samples}
         请用中文回答问题
         """),
        ("user","{text}")
    ])
    message_text=get_message_content(state["messages"][0])
    query=message_text
    embeddings=DashScopeEmbeddings(
        model=DashScopeEmbeddingModel
    )
    connection = "postgresql+psycopg://postgres:123456@localhost:5432/Couplet"  # Uses psycopg3!
    collection_name = "couplet"
    vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    )
    samples=[]
    scored_docs=vector_store.similarity_search(query,k=5)
    for doc in scored_docs:
        samples.append(doc.page_content)
    prompt=prompt_template.invoke({"text":query,"samples":"\n".join(samples)})
    writer({"couplet_prompt":prompt.messages[0].content})
    response=llm.invoke(prompt)
    return {"messages":[HumanMessage(content=response.content)],"type":"couplet"}
def other_node(state:State):
    print(">>> other_node")
    writer=get_stream_writer()
    writer({"node":">>> other_node"})
    return {"messages":[HumanMessage(content="other_node")],"type":"other"}

def routing_func(state:State):
    if state["type"]=="travel":
        return "travel_node"
    elif state["type"]=="joke":
        return "joke_node"
    elif state["type"]=="couplet":
        return "couplet_node"
    elif state["type"]=="END":
        return END
    else:
        return "other_node"
def get_message_content(message):
    """从消息对象中提取文本内容"""
    if isinstance(message, str):
        return message
    elif hasattr(message, 'content'):
        return message.content
    else:
        return str(message)
builder=StateGraph(State)
builder.add_node("supervisor_node",supervisor_node)
builder.add_node("travel_node",travel_node)
builder.add_node("joke_node",joke_node)
builder.add_node("couplet_node",couplet_node)
builder.add_node("other_node",other_node)


builder.add_edge(START,"supervisor_node")
builder.add_conditional_edges("supervisor_node",routing_func,["travel_node","joke_node","couplet_node","other_node",END])
builder.add_edge("travel_node","supervisor_node")
builder.add_edge("joke_node","supervisor_node")
builder.add_edge("couplet_node","supervisor_node")
builder.add_edge("other_node","supervisor_node")

checkPointer=InMemorySaver()
graph=builder.compile(checkpointer=checkPointer)

if __name__ == "__main__":
    config={
        "configurable":{
            "thread_id":"123"
        }
    }
    # for chunk in graph.stream({"messages":["请给我讲一个郭德纲的笑话"]},config=config,stream="custom"):
    #     print(chunk)
    # for chunk in graph.stream({"messages":["春回大地千山秀的下联是什么"]},config=config,stream="custom"):
    #     print(chunk)
    # for chunk in graph.stream({"messages":["我想要从西安到华山，请帮我做一个出行规划"]},config=config,stream="custom"):
    #     print(chunk)
    # for chunk in graph.stream({"messages":["你好啊"]},config=config,stream="custom"):
    #     print(chunk)
    async def main():
        async for chunk in graph.astream({"messages": ["我想要从西安到华山，请帮我做一个出行规划"]}, config=config, stream="custom"):
            print(chunk)
    
    asyncio.run(main())
