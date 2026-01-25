# main.py
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import requests
import json
from dotenv import load_dotenv
from collections import defaultdict
import time

# ✅ 导入数据库模块
from database import init_database, save_conversation, get_recent_conversations, get_stats, search_conversations

# --- 初始化应用 ---
app = FastAPI(title="Central AI Service")

# ✅ 应用启动时初始化数据库
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    init_database()
    print("✅ 数据库已初始化")

# --- 自定义 CORS 中间件（只允许 Shopify 域名）---
class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        
        # 检查是否是允许的域名
        allowed = (
            origin.endswith(".myshopify.com") or 
            origin == "https://admin.shopify.com" or
            origin.startswith("http://localhost") or  # 本地测试
            origin == "https://theqiflow.com" or      # ✅ 你的自定义域名 1
            origin == "https://fengshuisource.com"    # ✅ 你的自定义域名 2
        )
        
        # 处理预检请求（OPTIONS）
        if request.method == "OPTIONS":
            if allowed:
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type",
                        "Access-Control-Allow-Credentials": "true",
                    }
                )
            return Response(status_code=403)
        
        # 处理正常请求
        response = await call_next(request)
        
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

app.add_middleware(CustomCORSMiddleware)

# --- 速率限制配置 ---
# 每分钟请求追踪
minute_requests = defaultdict(lambda: {"count": 0, "reset_time": 0})
MINUTE_LIMIT = 10  # 每分钟最多 10 次

# 每日请求追踪
daily_requests = defaultdict(lambda: {"count": 0, "reset_time": 0})
DAILY_LIMIT = 100  # 每天最多 100 次

def get_client_ip(request: Request) -> str:
    """获取客户端真实 IP 地址"""
    # Render.com 会通过 X-Forwarded-For 传递真实 IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def check_rate_limit(ip: str) -> tuple[bool, str]:
    """
    检查速率限制
    返回: (是否允许请求, 错误信息)
    """
    now = time.time()
    
    # === 检查每分钟限制 ===
    minute_data = minute_requests[ip]
    
    # 如果超过 1 分钟，重置计数
    if now - minute_data["reset_time"] > 60:
        minute_data["count"] = 0
        minute_data["reset_time"] = now
    
    # 检查是否超过每分钟限制
    if minute_data["count"] >= MINUTE_LIMIT:
        return False, f"每分钟最多 {MINUTE_LIMIT} 次请求，请稍后再试"
    
    # === 检查每日限制 ===
    daily_data = daily_requests[ip]
    
    # 如果超过 24 小时，重置计数
    if now - daily_data["reset_time"] > 86400:
        daily_data["count"] = 0
        daily_data["reset_time"] = now
    
    # 检查是否超过每日限制
    if daily_data["count"] >= DAILY_LIMIT:
        return False, f"每天最多 {DAILY_LIMIT} 次请求，请明天再试"
    
    # === 通过检查，增加计数 ===
    minute_data["count"] += 1
    daily_data["count"] += 1
    
    return True, ""

# --- LLM 客户端 ---
class LLMClient:
    def __init__(self):
        load_dotenv()  # 加载环境变量
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.model = "google/gemini-3-flash-preview"
    
    def call_llm(self, prompt_text: str):
        """调用 OpenRouter API"""
        if not self.api_key:
            print("错误：未找到 OPENROUTER_API_KEY 环境变量")
            return "服务器配置错误，请联系管理员"
        
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
        
        except requests.exceptions.Timeout:
            print("错误：API 请求超时")
            return "请求超时，请稍后重试"
        
        except requests.exceptions.RequestException as e:
            print(f"错误：API 调用失败 - {e}")
            return "AI 服务暂时不可用，请稍后重试"
        
        except Exception as e:
            print(f"未知错误：{e}")
            return "服务出现问题，请稍后重试"

# 初始化 LLM 客户端
llm_client = LLMClient()

# --- API 数据模型 ---
class AIRequest(BaseModel):
    prompt: str

# --- API 端点 ---
@app.get("/")
def read_root():
    """健康检查端点"""
    return {
        "message": "AI Service 运行中",
        "status": "ok",
        "version": "1.0"
    }

@app.post("/api/ask-ai")
async def ask_ai_endpoint(request: AIRequest, req: Request):
    """
    AI 问答端点
    - 每分钟最多 10 次请求
    - 每天最多 100 次请求
    - ✅ 自动记录所有对话到数据库
    """
    # 获取客户端 IP
    client_ip = get_client_ip(req)
    
    # 检查速率限制
    allowed, error_msg = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=error_msg)
    
    # 记录开始时间
    start_time = time.time()
    
    # 调用 AI
    ai_response = llm_client.call_llm(request.prompt)
    
    # 计算响应时间
    response_time = time.time() - start_time
    
    # ✅ 保存到数据库
    save_conversation(
        ip=client_ip,
        prompt=request.prompt,
        response=ai_response,
        model=llm_client.model,
        response_time=response_time
    )
    
    # 打印使用情况日志
    print(f"[{client_ip}] 分钟: {minute_requests[client_ip]['count']}/{MINUTE_LIMIT} | 每日: {daily_requests[client_ip]['count']}/{DAILY_LIMIT}")
    
    return {"response": ai_response}

@app.get("/api/health")
def health_check():
    """健康检查（用于监控）"""
    return {"status": "healthy"}

# ✅ 新增：查看统计信息
@app.get("/admin/stats")
def get_usage_stats():
    """查看当前使用统计"""
    try:
        db_stats = get_stats()
    except:
        db_stats = {"error": "数据库暂无数据"}
    
    return {
        "database": db_stats,
        "rate_limits": {
            "minute_usage": {
                ip: data["count"] 
                for ip, data in minute_requests.items() 
                if data["count"] > 0
            },
            "daily_usage": {
                ip: data["count"] 
                for ip, data in daily_requests.items() 
                if data["count"] > 0
            }
        }
    }

# ✅ 新增：查看最近对话
@app.get("/admin/conversations")
def get_conversations(limit: int = 50):
    """查看最近的对话记录"""
    try:
        conversations = get_recent_conversations(limit)
        return {
            "total": len(conversations),
            "conversations": conversations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

# ✅ 新增：搜索对话
@app.get("/admin/search")
def search_conversations_endpoint(keyword: str = None, ip: str = None, limit: int = 50):
    """搜索对话记录"""
    try:
        results = search_conversations(keyword, ip, limit)
        return {
            "total": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
