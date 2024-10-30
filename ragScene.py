import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import numpy as np
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from vllm import LLM, SamplingParams
from FlagEmbedding import FlagReranker
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 环境变量和设备设置
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_folder = "/data2/lzy/cache/directory"

# 模型初始化
rope_scaling = {"type": "linear", "factor": 1.0}
bge_model = SentenceTransformer("BAAI/bge-large-zh", device=device, cache_folder=cache_folder)
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
MODEL_PATH = "/data2/lzy/py/ycc/Qwen/qwen/Qwen2-VL-2B-Instruct"
llm = LLM(model=MODEL_PATH, limit_mm_per_prompt={"image": 10, "video": 10}, device=device, rope_scaling=rope_scaling)
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# Elasticsearch连接
es = Elasticsearch("http://172.16.170.187:9200")

# 定义 State 类型
class State(TypedDict):
    messages: Annotated[List[str], add_messages]

# 创建 StateGraph 对象
graph_builder = StateGraph(State)

# 创建索引映射
def create_index_mappings(es, text_index, vector_index):
    text_mapping = {"properties": {"text": {"type": "text"}}}
    vector_mapping = {"properties": {"vector": {"type": "dense_vector", "dims": 768}}}
    for index, mapping in zip([text_index, vector_index], [text_mapping, vector_mapping]):
        if not es.indices.exists(index=index):
            es.indices.create(index=index, body={"mappings": mapping})

# 从多个 DOCX 文件提取文本内容
def extract_text_from_docx(file_paths):
    all_documents = []
    for file_path in file_paths:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

# 文本分块
def chunk_text(documents, chunk_size=500, overlap=100):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

# 嵌入文本块
def embed_chunks(chunks):
    embeddings = bge_model.encode(chunks, convert_to_tensor=True, device=device).cpu().numpy()
    if embeddings.shape[1] != 768:
        embeddings = embeddings[:, :768]
    return embeddings

# 获取当前索引的最大ID
def get_max_index_id(es, index_name):
    response = es.search(index=index_name, body={"query": {"match_all": {}}}, size=0)
    return response['hits']['total']['value']

# 将文本块嵌入并索引到 ES，增加start_id参数
def index_chunks(es, chunks, text_index="text-index", vector_index="vector-index", start_id=0):
    for idx, chunk in enumerate(chunks):
        vector = embed_chunks([chunk])[0].tolist()
        es.index(index=text_index, id=start_id + idx, body={"text": chunk})
        es.index(index=vector_index, id=start_id + idx, body={"vector": vector})

# 根据向量相似度检索top-k个相关块
def retrieve_top_k(es, query, k=10, text_index="text-index", vector_index="vector-index"):
    query_vector = embed_chunks([query])[0].tolist()
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    response = es.search(index=vector_index, body={"query": script_query, "size": k})
    hits = response['hits']['hits']
    ids = [hit['_id'] for hit in hits]
    return [es.get(index=text_index, id=doc_id)['_source']['text'] for doc_id in ids]

# 重排序
def rerank(chunks, query):
    scores = reranker.compute_score([[query, chunk] for chunk in chunks])
    scores = np.array(scores)
    reranked_indices = np.argsort(-scores)
    return [chunks[i] for i in reranked_indices[:3]]

# 生成答案，加入对话历史上下文
def generate_answer(context, query, user_messages, assistant_messages):
    # 将对话历史整合成一个提示
    history_prompt = "\n".join(
        [f"用户：{user_msg}\n助手：{assistant_msg}"
         for user_msg, assistant_msg in zip(user_messages, assistant_messages)]
    )
    prompt = f"""
    以下是用户与助手的对话历史。助手需根据历史对话、提供的上下文和当前用户问题进行回答，确保回答连贯、准确，并避免重复内容。
    【对话历史】
    {history_prompt}
    【背景信息】
    以下是相关背景信息和参考内容，可能包含用户问题的答案或帮助解决问题的线索：
    {context}
    【用户当前问题】
    用户的问题：{query}
    【助手回答要求】
    - 请参考对话历史和背景信息回答用户的问题
    - 回答应直接、简洁，不需重复问题
    - 如背景信息未覆盖用户问题，结合已知信息给予合理推测或指引
    助手：
    """
    response = llm.generate([{"prompt": prompt}], sampling_params=sampling_params)
    return response[0].outputs[0].text.strip()

# 判断用户问题与文档内容的相关性   意图识别
def is_relevant_to_documents(query, doc_chunks, threshold=0.1):
    # 计算相关性分数
    relevance_scores = reranker.compute_score([[query, chunk] for chunk in doc_chunks])
    # 打印每个文档的分数
    # print(f"分数: {relevance_scores}")  # 输出每个分数
    # 检查是否有任何分数高于阈值
    for score in relevance_scores:
        if score >= threshold:
            return True
    return False  # 如果没有分数高于阈值，返回 False


# 定义全局缓存变量
base_chunks_cache = None
user_chunks_cache = None
user_files_cache = None
base_files_cache = None

# RAG主流程函数中引用有记忆对话
def rag_pipeline(user_query, user_messages, assistant_messages, base_file_paths=None):
    global base_chunks_cache, user_chunks_cache, user_files_cache, base_files_cache

    # 定义临时索引名称
    temp_text_index = "temp-text-index"
    temp_vector_index = "temp-vector-index"

    # 用户文件的处理
    user_file_paths = ["/data2/lzy/static/用户上传的文件.docx"]
    if user_file_paths != user_files_cache:
        create_index_mappings(es, temp_text_index, temp_vector_index)
        user_text = extract_text_from_docx(user_file_paths)
        user_chunks = chunk_text(user_text)
        current_max_id = get_max_index_id(es, temp_text_index) if es.indices.exists(temp_text_index) else 0
        index_chunks(es, user_chunks, text_index=temp_text_index, vector_index=temp_vector_index, start_id=current_max_id + 1)
        user_chunks_cache = user_chunks
        user_files_cache = user_file_paths

    # 用户文件处理完成，获取与用户查询相关的内容
    top_k_chunks = retrieve_top_k(es, user_query, text_index=temp_text_index, vector_index=temp_vector_index) if user_chunks_cache else []
    
    # 基础文件的处理
    if base_chunks_cache is None or base_file_paths != base_files_cache:
        base_text = extract_text_from_docx(base_file_paths)
        base_chunks = chunk_text(base_text)
        create_index_mappings(es, "text-index", "vector-index")
        current_max_id = get_max_index_id(es, "text-index") if es.indices.exists("text-index") else 0
        index_chunks(es, base_chunks, text_index="text-index", vector_index="vector-index", start_id=current_max_id + 1)
        base_chunks_cache = base_chunks
        base_files_cache = base_file_paths

    if not top_k_chunks:
        top_k_chunks = retrieve_top_k(es, user_query)

    # 检查相关性
    relevant = is_relevant_to_documents(user_query, base_chunks_cache + user_chunks_cache)
    if not relevant:
        return generate_answer("", user_query, user_messages, assistant_messages)

    # 重排序并生成回答
    reranked_chunks = rerank(top_k_chunks, user_query)
    context = " ".join(reranked_chunks)
    return generate_answer(context, user_query, user_messages, assistant_messages)

# 定义用户和助手的消息记录
user_messages = []
assistant_messages = []

# Chatbot 主函数，增加对历史上下文的引用
def chatbot(user_messages, assistant_messages):
    if user_messages:
        user_message_content = user_messages[-1]
        answer = rag_pipeline(
            user_message_content,
            user_messages,
            assistant_messages,
            base_file_paths=[
                "/data2/lzy/static/立项用地阶段窗口办事指南审查要点.docx",
                "/data2/lzy/static/审查要点.docx"
            ]
        )
        assistant_messages.append(answer)
        print("Assistant:\n", answer)

# 对话循环
while True:
    user_input = input("User: ")
    user_messages.append(user_input)
    chatbot(user_messages, assistant_messages)
    print("\n")
