#实体识别
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageEnhance, ImageFilter
from starlette.responses import JSONResponse
from io import BytesIO
import torch
import uvicorn
import re  # 导入正则表达式模块

app = FastAPI()

class RequestData(BaseModel):
    text: str = ""

MODEL_PATH = "/data2/lzy/py/ycc/Qwen/qwen/Qwen2-VL-2B-Instruct"

rope_scaling = {
    "type": "linear",
    "factor": 1.0  # 根据你的模型和需求设置合适的值
}

# 初始化LLM模型
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
    rope_scaling=rope_scaling  # 添加 rope_scaling 参数
)

sampling_params = SamplingParams(
    temperature=0.7,  # 增加温度以增加多样性
    top_p=0.9,
    repetition_penalty=1.2,  # 增加重复惩罚
    max_tokens=256,  # 减少最大生成长度
    stop_token_ids=[],
)

# 初始化处理器和模型
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH)

required_fields = [
        "项目代码", "单位名称", "法人代表联系电话", "委托代理人","邮箱",
        "单位地址", "项目名称", "审批结果送达方式","印章信息"
    ]

def format_extracted_text(extracted_texts):
    """ 将提取的文本格式化为 '字段名：值' 的形式 """
    
    extracted_info = {field: None for field in required_fields}
    formatted_texts = []

    # 解析提取的文本
    for line in extracted_texts.split('\n'):
        for field in required_fields:
            if field in line:
                value = line.split(field, 1)[1].strip().strip('：')
                print(value)
                extracted_info[field] = value

    # 格式化提取的信息
    for field, value in extracted_info.items():
        if value:
            formatted_texts.append(f"{field}: {value}")

    return formatted_texts, extracted_info

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
    """ 处理上传的图片和文本，生成响应 """
    # 解析上传的图片
    image = None
    if image_file:
        image_bytes = await image_file.read()
        image = Image.open(BytesIO(image_bytes))
        image_path = f"/tmp/{image_file.filename}"  # 临时保存图片
        image.save(image_path)
    else:
        raise HTTPException(status_code=400, detail="No image file uploaded")


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
    extracted_texts = outputs[0].outputs[0].text
    generated_text=extracted_texts
    print("Extracted Texts:", extracted_texts)

    # 格式化提取的文本
    formatted_texts, extracted_info = format_extracted_text(extracted_texts)
    print("Formatted Texts:", formatted_texts)
    print("Extracted Info:", extracted_info)

    # 验证申请表内容
    validation_errors, validated_info = validate_application_form(extracted_info, project_info)
    print("Validation Errors:", validation_errors)
    print("Validated Info:", validated_info)
    

    # 打印生成的每一步输出，检查是否有重复的内容生成
    print("Generated Text Steps:")
    for output in outputs[0].outputs:
        print(output.text)

    # 返回结果
    response_data = {
        "generated_text": generated_text,
        "validation_errors": validation_errors,
        "extracted_info": validated_info,
        "all_extracted_texts": extracted_texts  # 包含所有提取到的文本
    }

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)