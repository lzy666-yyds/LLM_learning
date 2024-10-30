from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.llms import Tongyi
import os

# 设置环境变量，确保替换为您自己的API密钥
os.environ["DASHSCOPE_API_KEY"] = 'sk-25847a1cbf934d068a0a4e87ed4e75e0'

# 定义State类型，其中messages字段是一个列表，使用add_messages函数进行更新
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 创建StateGraph对象，使用定义的State类型作为参数
graph_builder = StateGraph(State)

# 创建通义千问plus实例，使用特定的模型
llm = Tongyi(model="qwen-plus")

# 定义聊天机器人函数
def chatbot(state: State):
    # 使用llm的invoke方法处理输入消息，并返回处理后的消息列表
    # 假设state["messages"]中存储的是用户的最后一条消息
    if state["messages"]:
        user_message = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else state["messages"][-1]  # 获取消息内容
        return {"messages": [llm.invoke(user_message)]}  # 调用模型并传递消息内容
    else:
        return {"messages": []}  # 如果没有消息，返回空列表

# 添加节点到图构建器
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# 定义流式处理图更新的函数
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            # 直接打印整个消息字符串
            if value["messages"]:
                message = value["messages"][-1]
                if hasattr(message, 'content'):
                    print("Assistant:", message.content)  # 打印助理的响应内容
                else:
                    print("Assistant:", message)  # 如果是字符串，则直接打印

# 主循环，不断获取用户输入并处理
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # 如果input()不可用，则进行回退
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break