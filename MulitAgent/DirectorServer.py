import random
import gradio as gr
from MultiAgent import graph
import asyncio

async def process_input_async(text):
    config ={
        "configurable":{
            "thread_id":random.randint(0,1000000)
        }
    }
    result=await graph.ainvoke({"messages":[text]},config=config)
    return result["messages"][-1].content
def process_input(text):
    # 在同步函数中运行异步代码
    return asyncio.run(process_input_async(text))

with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Multi-Agent")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 可以规划路线、对对联、讲笑话，快来试试吧。")
            input_text=gr.Textbox(label="问题",placeholder="请输入问题",value="请给我讲一个郭德纲的笑话")
            submit_btn=gr.Button("提交",variant="primary")
        with gr.Column():
            output_text=gr.Textbox(label="输出")
    submit_btn.click(process_input,inputs=[input_text],outputs=[output_text])

demo.launch(server_name="0.0.0.0", server_port=7999)