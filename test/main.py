import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
#from pydantic import SecretStr

load_dotenv()

modelUrl=(os.environ["ModelUrl"])
modelKey=(os.environ["ModelKey"])
modelName=(os.environ["ModelName"])

print(modelUrl)
print(modelName)
print(modelKey)

llm=ChatOpenAI(
    base_url=modelUrl,
    model=modelName,
    api_key=modelKey
)
# 直接调用大模型
print(llm.invoke("你是谁").content)

#智能体

agent=create_agent(llm,system_prompt="你叫慕言，是我的专属学习伙伴")
msg={"message":[
        {
            "role":"user",
            "content":"你是谁"
         }
        ]}
print(agent.invoke(msg)["messages"])