# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import json
from dotenv import load_dotenv

# --- 1. 初始化应用和配置 ---
app = FastAPI(title="Central AI Service")

# 配置CORS，允许所有来源的请求，方便您在任何网站上调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LLM客户端逻辑 ---
class LLMClient:
    def __init__(self):
        load_dotenv() # 加载 .env 文件 (仅本地测试时有效)
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.model = "google/gemini-2.5-flash" # 使用您指定的模型

    def call_llm(self, prompt_text: str):
        if not self.api_key:
            print("ERROR: API Key not found!")
            return "错误：服务器未正确配置API密钥。"
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt_text}]
                }),
                timeout=180
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"ERROR calling API: {e}")
            return f"API调用时发生错误: {str(e)}"

llm_client = LLMClient()

# --- 3. 定义API的输入和输出 ---
class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "AI Service is running."}

@app.post("/api/ask-ai")
def ask_ai_endpoint(request: AIRequest):
    ai_response = llm_client.call_llm(request.prompt)
    return {"response": ai_response}