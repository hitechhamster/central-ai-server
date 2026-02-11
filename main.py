# main.py
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import os
import httpx
import json
from dotenv import load_dotenv
from collections import defaultdict
import time
from typing import Optional

# ✅ 导入数据库模块
from database import init_database, save_conversation, get_recent_conversations, get_stats, search_conversations

# --- Lifespan 管理（替代弃用的 on_event）---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    init_database()
    print("✅ 数据库已初始化")
    print("✅ AI 服务启动成功")
    yield
    # 关闭时
    print("👋 AI 服务关闭")

# --- 初始化应用 ---
app = FastAPI(title="Central AI Service", lifespan=lifespan)

# --- 自定义 CORS 中间件（只允许 Shopify 域名）---
class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        
        # 检查环境
        is_production = os.environ.get("ENVIRONMENT") == "production"
        
        # 检查是否是允许的域名
        allowed = (
            origin.endswith(".myshopify.com") or 
            origin == "https://admin.shopify.com" or
            origin == "https://theqiflow.com" or
            origin == "https://fengshuisource.com" or
            (not is_production and origin.startswith("http://localhost"))
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
minute_requests = defaultdict(lambda: {"count": 0, "reset_time": 0})
MINUTE_LIMIT = 10

daily_requests = defaultdict(lambda: {"count": 0, "reset_time": 0})
DAILY_LIMIT = 100

def get_client_ip(request: Request) -> str:
    """获取客户端真实 IP 地址"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    return request.client.host

def check_rate_limit(ip: str):
    """检查速率限制，返回: (是否允许请求, 错误信息)"""
    now = time.time()
    
    # 检查每分钟限制
    minute_data = minute_requests[ip]
    if now - minute_data["reset_time"] > 60:
        minute_data["count"] = 0
        minute_data["reset_time"] = now
    
    if minute_data["count"] >= MINUTE_LIMIT:
        return False, f"每分钟最多 {MINUTE_LIMIT} 次请求，请稍后再试"
    
    # 检查每日限制
    daily_data = daily_requests[ip]
    if now - daily_data["reset_time"] > 86400:
        daily_data["count"] = 0
        daily_data["reset_time"] = now
    
    if daily_data["count"] >= DAILY_LIMIT:
        return False, f"每天最多 {DAILY_LIMIT} 次请求，请明天再试"
    
    # 通过检查，增加计数
    minute_data["count"] += 1
    daily_data["count"] += 1
    
    return True, ""

# --- 异步 LLM 客户端 ---
class AsyncLLMClient:
    """异步 LLM 客户端，支持多模型"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.default_model = "google/gemini-3-flash-preview"
        
        # ✅ 允许的模型白名单
        self.allowed_models = {
            "google/gemini-3-flash-preview",   # 默认快速模型
            "google/gemini-3-pro-preview",     # 高级推理模型（用于复杂任务）
        }
    
    def get_model(self, requested_model: Optional[str]) -> str:
        """验证并返回要使用的模型"""
        if requested_model and requested_model in self.allowed_models:
            return requested_model
        return self.default_model
    
    async def call_llm(self, prompt_text: str, model: Optional[str] = None) -> str:
        """异步调用 OpenRouter API"""
        if not self.api_key:
            print("❌ 错误：未找到 OPENROUTER_API_KEY 环境变量")
            return "服务器配置错误，请联系管理员"
        
        use_model = self.get_model(model)
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://theqiflow.com",
                        "X-Title": "Feng Shui AI Service"
                    },
                    json={
                        "model": use_model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "temperature": 0.7,
                        "max_tokens": 8000
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
        
        except httpx.TimeoutException:
            print("⏱️ 错误：API 请求超时")
            return "请求超时，请稍后重试"
        
        except httpx.HTTPStatusError as e:
            print(f"🚫 HTTP 错误：{e.response.status_code} - {e.response.text}")
            return "AI 服务暂时不可用，请稍后重试"
        
        except Exception as e:
            print(f"❌ 未知错误：{e}")
            return "服务出现问题，请稍后重试"

# 初始化异步 LLM 客户端
llm_client = AsyncLLMClient()

# --- API 数据模型 ---
class AIRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    model: Optional[str] = None  # ✅ 可选：指定使用的模型

# --- API 端点 ---
@app.get("/")
def read_root():
    """健康检查端点"""
    return {
        "message": "AI Service 运行中",
        "status": "ok",
        "version": "2.0",
        "default_model": llm_client.default_model,
        "available_models": list(llm_client.allowed_models)
    }

@app.post("/api/ask-ai")
async def ask_ai_endpoint(request: AIRequest, req: Request):
    """
    AI 问答端点（异步版本）
    
    参数：
    - prompt: 问题内容
    - model: 可选，指定模型 (如 "google/gemini-2.5-pro-preview")
    """
    # 获取客户端 IP
    client_ip = get_client_ip(req)
    
    # 检查速率限制
    allowed, error_msg = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=error_msg)
    
    # 确定使用的模型
    use_model = llm_client.get_model(request.model)
    
    # 记录开始时间
    start_time = time.time()
    
    # ✅ 异步调用 AI
    ai_response = await llm_client.call_llm(request.prompt, use_model)
    
    # 计算响应时间
    response_time = time.time() - start_time
    
    # 保存到数据库（容错处理）
    try:
        save_conversation(
            ip=client_ip,
            prompt=request.prompt,
            response=ai_response,
            model=use_model,
            response_time=response_time
        )
    except Exception as e:
        print(f"⚠️ 保存对话失败: {e}")
    
    # 打印使用情况日志
    print(f"[{client_ip}] 模型: {use_model} | 耗时: {response_time:.2f}s | 分钟: {minute_requests[client_ip]['count']}/{MINUTE_LIMIT}")
    
    return {"response": ai_response, "model": use_model}

@app.get("/api/health")
def health_check():
    """健康检查（用于监控）"""
    return {"status": "healthy", "timestamp": time.time()}

# --- 管理端点（暂不加认证）---
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
