import base64
import urllib.parse
import requests
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
from io import BytesIO
import torch
import uvicorn
from starlette.responses import JSONResponse
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app = FastAPI()

class RequestData(BaseModel):
    text: str = ""

MODEL_PATH = "/data2/lzy/py/ycc/Qwen/qwen/Qwen2-VL-2B-Instruct"

rope_scaling = {
    "type": "linear",
    "factor": 1.0  # 根据你的模型和需求设置合适的值
}

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
    rope_scaling=rope_scaling  # 添加 rope_scaling 参数
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=512,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

required_fields = [
    "项目代码", "单位名称", "法人代表联系电话", "委托代理人", "证件号码(身份证或台胞证)",
    "联系电话", "邮箱", "组织机构代码证号或统一社会信用代码证号", "单位地址", "委托日期", "项目名称", "印章信息",
]

# 获取百度AI的access token
def get_access_token():
    api_key = "s2Xlot59KoaugFoVd0uixMFR"
    secret_key = "WN1gSGyoqvfvTVDe3ceOukUAjiSrbEMs"
    auth_url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    response = requests.post(auth_url, params=params)
    return response.json()["access_token"]

# 调用百度AI OCR接口识别印章信息
def detect_seal(image_bytes):
    if not image_bytes:
        print("Image bytes are empty")
        return {}

    access_token = get_access_token()
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/seal"
    # 将图像字节转换为 Base64 编码的字符串
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    params = {"image": img_base64}
    print("Base64 encoded image:", params)  # 确保图像已正确编码
    request_url = request_url + "?access_token=" + access_token
    print(request_url)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    response = requests.request("POST", request_url, headers=headers, data=params)
    print("Response status code:", response.status_code)  # 检查响应状态码
    print("Response content:", response.content)  # 检查响应内容
    return response.json()

# 提取文本
def format_extracted_text(extracted_texts):
    extracted_info = {field: None for field in required_fields}

    # 解析提取的文本
    for line in extracted_texts.split('\n'):
        for field in required_fields:
            if field in line:
                value = line.split(field, 1)[1].strip().strip('：')
                print(value)
                extracted_info[field] = value
    return extracted_info

# 校验身份证
def validate_id_number(id_number):
    id_pattern = r"^\d{17}[0-9X]$"
    taiwan_id_pattern = r"^\d{8}$"

    if re.match(id_pattern, id_number) or re.match(taiwan_id_pattern, id_number):
        return True
    return False

def validate_application_form(extracted_info, project_info):
    """ 验证申请表内容 """
    errors = []

    # 检查必填项是否填写
    for field in required_fields:
        if not extracted_info.get(field):
            errors.append(f"缺少{field}的填写。")

    # 检查联系电话格式
    if extracted_info.get("法人代表联系电话"):
        phone_number = extracted_info["法人代表联系电话"]
        if not re.match(r"^\d{11}$", phone_number):
            errors.append("联系电话格式不正确，应为11位数字。")

    # 检查证件号码格式
    if extracted_info.get("证件号码(身份证或台胞证)"):
        id_number = extracted_info["证件号码(身份证或台胞证)"]
        if not validate_id_number(id_number):
            errors.append("证件号码格式不正确，身份证应为18位数字，最后一位可以是数字或大写的X；台胞证应为8位数字。")

    # 检查项目代码和单位名称是否一致
    if project_info:
        if project_info.get("project_code") != extracted_info.get("项目代码"):
            errors.append("项目代码与市建管系统中的项目基本信息不一致。")
        if project_info.get("unit_name") != extracted_info.get("单位名称"):
            errors.append("单位名称与市建管系统中的项目基本信息不一致。")

    return errors, extracted_info

# 模拟从外部系统获取的项目信息
project_info = {
    "project_code": "2306-350205-06-01-114548",
    "unit_name": "厦门市海沧区教育局"
}

@app.post("/generate/")
async def create_item(text: str = Form(...), image_file: UploadFile = File(None)):
    image = None
    seal_detection_result = None
    if image_file:
        image_bytes = await image_file.read()
        print(f"Image bytes length: {len(image_bytes)}")  # 检查图像字节长度
        if not image_bytes:
            print("Image bytes are empty")
            return JSONResponse(content={"error": "Image file is empty"})

        image = Image.open(BytesIO(image_bytes))

        # 保存 image_bytes 后重置指针
        image_bytes_io = BytesIO(image_bytes)
        image_bytes_io.seek(0)

        # 调用 detect_seal 函数进行印章检测
        seal_detection_result = detect_seal(image_bytes_io.read())
        print("Seal detection result:", seal_detection_result)

    # 构建输入消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} if image else {},
                {"type": "text", "text": text},
            ],
        },
    ]

    # 应用聊天模板
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 处理视觉信息
    image_inputs, _ = process_vision_info(messages)

    # 准备多模态数据
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    # 生成输入
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    # 生成输出
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    extracted_info = format_extracted_text(generated_text)

    # 提取印章的主要文字
    if seal_detection_result and "result" in seal_detection_result and seal_detection_result["result"]:
        try:
            seal_major_text = seal_detection_result["result"][0]["major"]["words"]
            extracted_info["印章信息"] = seal_major_text
        except (KeyError, IndexError) as e:
            print(f"Error extracting seal information: {e}")

    # 验证申请表内容
    validation_errors, validated_info = validate_application_form(extracted_info, project_info)
    # 返回结果
    response_data = {
        "generated_text": generated_text,
        "validation_errors": validation_errors,
        "extracted_info": validated_info,
    }

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)