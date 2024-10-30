import torch
from modelscope import snapshot_download
 
# snapshot_download函数用于下载模型
model_dir = snapshot_download(
    'qwen/Qwen2-VL-2B-Instruct',  # 模型名称
    cache_dir='/data2/lzy/py/ycc/Qwen/',  # 缓存目录
    revision='master'  # 版本号
    )