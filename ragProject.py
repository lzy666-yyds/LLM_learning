import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
from typing import Annotated, List
from typing_extensions import TypedDict
import numpy as np
import faiss
import torch
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from FlagEmbedding import FlagReranker

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用第3个GPU
cache_folder = "/data2/lzy/cache/directory"

# 检测并使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化句子嵌入模型
bge_model = SentenceTransformer("BAAI/bge-large-zh", device=device, cache_folder=cache_folder)
MODEL_PATH = "/data2/lzy/py/ycc/Qwen/qwen/Qwen2-VL-2B-Instruct"
# 初始化重排器
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

# 初始化大语言模型的参数
rope_scaling = {
    "type": "linear",
    "factor": 1.0
}
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
    device=device,
    rope_scaling=rope_scaling
)

# 设定采样参数
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# 定义State类型
class State(TypedDict):
    messages: Annotated[List[str], add_messages]  # 使用列表来存储消息历史

# 创建StateGraph对象，使用定义的State类型作为参数
graph_builder = StateGraph(State)

# 全局变量，用于保存FAISS索引和文档块
faiss_index = None
all_chunks = []  # 存储所有文档块内容
existing_files = set()  # 追踪已处理的文件

# 文档处理和嵌入函数
def extract_text_from_docx(file_path):
    """从 DOCX 文件中提取文本内容"""
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    return documents

def chunk_text(documents, chunk_size=500, overlap=100):
    """将文档分块，以便进行嵌入和检索"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def embed_chunks(chunks):
    """将文本块嵌入为向量表示"""
    return bge_model.encode(chunks, convert_to_tensor=True, device=device).cpu().numpy()

def load_faiss_index(dimension):
    """加载现有的 FAISS 索引，或创建新的索引"""
    global existing_files, all_chunks
    try:
        index = faiss.read_index("/data2/lzy/static/faiss_index.index")
        print("加载现有的FAISS索引。")
        # 加载已处理的文件列表
        with open("/data2/lzy/static/existing_files.pkl", "rb") as f:
            existing_files = pickle.load(f)
        # 加载所有文档块
        with open("/data2/lzy/static/all_chunks.pkl", "rb") as f:
            all_chunks = pickle.load(f)
    except Exception as e:
        print("未找到现有的FAISS索引，创建一个新的。")
        index = faiss.IndexFlatL2(dimension)  # 创建新的 FAISS 索引
    return index

def retrieve_top_k(index, query, k=100):
    """根据查询获取最相关的 K 个向量的索引"""
    query_vector = embed_chunks([query])  # 嵌入查询文本
    distances, indices = index.search(query_vector, k)  # 使用 FAISS 检索
    return indices.flatten()  # 返回扁平化的索引

def rerank(chunks, query):
    """对检索到的文档块进行重排序，以提高相关性"""
    scores = reranker.compute_score([[query, chunk] for chunk in chunks])  # 计算分数
    scores = np.array(scores)  # 转换为 NumPy 数组
    reranked_indices = np.argsort(-scores)  # 按分数排序
    return [chunks[i] for i in reranked_indices[:10]]  # 返回前10个块

def generate_answer(context, query):
    """生成针对用户查询的回答"""
    prompt = f"""
    上下文：根据以下官方文件和指南，您可以获得所需的申请材料和填写要求：
    {context}

    用户问题：{query}

    助手：请提供详细的申请材料清单及每项材料的填写要求，以便用户正确填写申请。
    """
    response = llm.generate([{"prompt": prompt}], sampling_params=sampling_params)  # 使用 LLM 生成回答
    return response[0].outputs[0].text.strip()  # 返回生成的文本

def chatbot(state: State):
    """处理用户输入并生成相应的回答"""
    if state["messages"]:
        #print(state["messages"])    这个state["messages"]每次都被覆盖掉了，要插入问题和用户的回答
        user_message = state["messages"][-1]  # 获取用户最新消息
        if hasattr(user_message, 'content'):
            user_message_content = user_message.content  # 提取消息内容
        else:
            user_message_content = user_message  # 如果是字符串，直接使用

        answer = rag_pipeline(user_message_content)  # 使用 RAG 流水线生成答案
        # 将用户消息和助手回答添加到消息历史
        state["messages"].append(answer)
        return {"messages": [answer]}  # 返回助手的响应
    else:
        return {"messages": []}  # 如果没有消息，返回空列表

def save_faiss_index(index):
    """保存 FAISS 索引和文档块信息"""
    faiss.write_index(index, "/data2/lzy/static/faiss_index.index")  # 保存索引
    with open("/data2/lzy/static/existing_files.pkl", "wb") as f:
        pickle.dump(existing_files, f)  # 保存已处理的文件列表
    with open("/data2/lzy/static/all_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)  # 保存所有文档块

def update_faiss_index(index, embeddings):
    """将新的嵌入添加到 FAISS 索引中并保存"""
    index.add(embeddings)  # 添加嵌入
    save_faiss_index(index)  # 保存索引
    return index

def rag_pipeline(user_query):
    global faiss_index, existing_files, all_chunks
    #要让他如果用户上传了特定的文件 就要用临时的文件  只对这几个文件进行搜索
    file_paths = [
        "/data2/lzy/static/立项用地阶段窗口办事指南审查要点.docx",
        "/data2/lzy/static/审查要点.docx",
        # 在这里添加更多文件路径
    ]

    new_files = [file for file in file_paths if file not in existing_files]  # 检查是否有新文件

    if new_files:
        # 处理新文件
        for file_path in new_files:
            documents = extract_text_from_docx(file_path)  # 提取文本
            chunks = chunk_text(documents)  # 切分文本
            all_chunks.extend(chunks)  # 累加新的文档块
            existing_files.add(file_path)  # 添加到已处理文件列表

        new_embeddings = embed_chunks(all_chunks[-len(chunks):])  # 仅对新增的块嵌入
        dimension = new_embeddings.shape[1]  # 获取嵌入的维度
        faiss_index = load_faiss_index(dimension)  # 加载或创建 FAISS 索引
        faiss_index = update_faiss_index(faiss_index, new_embeddings)  # 更新索引
    else:
        print("没有新文件添加到向量库中。")

    # 5. 检索top k块
    top_k_indices = retrieve_top_k(faiss_index, user_query)  # 获取相关块的索引
    top_k_chunks = [all_chunks[i] for i in top_k_indices]  # 根据索引获取块内容
    # 6. 对检索到的块进行重排序
    reranked_chunks = rerank(top_k_chunks, user_query)  # 重排序
    # 7. 生成最终答案
    context = " ".join(reranked_chunks)  # 拼接上下文
    return generate_answer(context, user_query)  # 返回生成的答案

# 添加节点到图构建器
graph_builder.add_node("chatbot", chatbot)  # 添加聊天机器人节点
graph_builder.add_edge(START, "chatbot")  # 定义起始节点
graph_builder.add_edge("chatbot", END)  # 定义结束节点
graph = graph_builder.compile()  # 编译图

def stream_graph_updates(user_input: str):
    """流式处理用户输入并更新图"""
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if value["messages"]:
                message = value["messages"][-1]  # 获取最新消息
                print("助手:", message)  # 打印助手回复

# 主循环，不断获取用户输入并处理
while True:
    try:
        user_input = input("用户: ")  # 获取用户输入
        if user_input.lower() in ["quit", "exit", "q"]:  # 退出条件
            print("再见！")
            break
        stream_graph_updates(user_input)  # 处理用户输入
    except Exception as e:
        print("错误:", e)  # 捕获并打印错误信息
        break


#后续实现存入es  进行读取
