# main.py
from fastapi import FastAPI, HTTPException, Request, Response, Form
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
from datetime import datetime, timezone
from urllib.parse import parse_qs

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
        self.api_key = os.environ.get('GOOGLE_GEMINI_API_KEY')
        self.default_model = "gemini-3.1-flash-lite-preview"
        
        # ✅ 允许的模型白名单
        self.allowed_models = {
            "gemini-3-flash-preview",   # 默认快速模型
            "gemini-3-pro-preview",     # 高级推理模型（用于复杂任务）
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
        "version": "2.1",
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


# ════════════════════════════════════════════════════════════════
# ✅ GUMROAD WEBHOOK → GOOGLE ADS 离线转化上传
# ════════════════════════════════════════════════════════════════

async def get_google_access_token() -> Optional[str]:
    """用 refresh_token 换取 Google OAuth2 access_token"""
    client_id = os.environ.get("GOOGLE_ADS_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_ADS_CLIENT_SECRET")
    refresh_token = os.environ.get("GOOGLE_ADS_REFRESH_TOKEN")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Google Ads OAuth 凭证缺失")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token"
                }
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
    except Exception as e:
        print(f"❌ 获取 Google access_token 失败: {e}")
        return None


async def upload_offline_conversion(gclid: str, sale_id: str, amount: float, sale_time: str):
    """
    上传离线转化到 Google Ads API
    
    参数:
    - gclid: Google Click ID
    - sale_id: Gumroad 订单 ID（用作 transaction_id 去重）
    - amount: 购买金额
    - sale_time: 购买时间 (格式: "2025-01-15 12:30:00+00:00")
    """
    customer_id = os.environ.get("GOOGLE_ADS_CUSTOMER_ID")          # 10位数字，无横线
    developer_token = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN")
    conversion_action_id = os.environ.get("GOOGLE_ADS_CONVERSION_ACTION_ID")  # 转化操作 ID
    
    if not all([customer_id, developer_token, conversion_action_id]):
        print("❌ Google Ads 配置缺失，跳过离线转化上传")
        print(f"   customer_id: {'✅' if customer_id else '❌'}")
        print(f"   developer_token: {'✅' if developer_token else '❌'}")
        print(f"   conversion_action_id: {'✅' if conversion_action_id else '❌'}")
        return False
    
    access_token = await get_google_access_token()
    if not access_token:
        return False
    
    # 构建转化数据
    conversion_payload = {
        "conversions": [
            {
                "conversionAction": f"customers/{customer_id}/conversionActions/{conversion_action_id}",
                "gclid": gclid,
                "conversionValue": amount,
                "conversionDateTime": sale_time,
                "currencyCode": "USD",
                "orderId": sale_id,
                "consent": {
                    "adUserData": "GRANTED"
                }
            }
        ],
        "partialFailure": True
    }
    
    api_url = f"https://googleads.googleapis.com/v19/customers/{customer_id}:uploadClickConversions"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "developer-token": developer_token,
        "Content-Type": "application/json"
    }
    
    # 如果使用 MCC（经理账户），需要加 login-customer-id
    login_customer_id = os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if login_customer_id:
        headers["login-customer-id"] = login_customer_id
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(api_url, headers=headers, json=conversion_payload)
            result = resp.json()
            
            if resp.status_code == 200:
                # 检查 partial failure
                if result.get("partialFailureError"):
                    print(f"⚠️ Google Ads 部分失败: {json.dumps(result['partialFailureError'], indent=2)}")
                    return False
                else:
                    print(f"✅ Google Ads 离线转化上传成功! gclid={gclid[:20]}... sale_id={sale_id}")
                    return True
            else:
                print(f"❌ Google Ads API 错误 {resp.status_code}: {json.dumps(result, indent=2)}")
                return False
                
    except Exception as e:
        print(f"❌ Google Ads API 调用异常: {e}")
        return False


@app.post("/api/gumroad-webhook")
async def gumroad_webhook(request: Request):
    """
    Gumroad Ping Webhook 端点
    
    Gumroad 购买成功后会 POST 到这里（x-www-form-urlencoded 格式）
    包含: email, sale_id, product_id, price, url_params 等
    
    流程:
    1. 解析 Gumroad POST 数据
    2. 从 url_params 或 custom_fields 提取 gclid
    3. 调用 Google Ads API 上传离线转化
    """
    try:
        # Gumroad 发送 x-www-form-urlencoded 格式
        body = await request.body()
        body_text = body.decode("utf-8")
        form_data = dict(parse_qs(body_text, keep_blank_values=True))
        
        # parse_qs 返回的值是列表，取第一个
        data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in form_data.items()}
        
        # 提取关键字段
        sale_id = data.get("sale_id", "")
        email = data.get("email", "")
        product_id = data.get("product_id", "")
        product_name = data.get("product_name", "")
        price = data.get("price", "0")           # 单位: 分 (cents)
        seller_id = data.get("seller_id", "")
        is_test = data.get("test", "false")
        url_params = data.get("url_params", "")   # Gumroad 会传递购买链接上的 URL 参数
        
        # 价格转换（Gumroad 以分为单位）
        try:
            amount_cents = int(price)
            amount_usd = amount_cents / 100.0
        except (ValueError, TypeError):
            amount_usd = 36.0  # 默认值
        
        # ════════════════════════════════════════
        # 提取 gclid（从多个可能的位置）
        # ════════════════════════════════════════
        gclid = None
        
        # 方法1: 从 url_params 提取
        if url_params:
            try:
                parsed_params = parse_qs(url_params)
                if "gclid" in parsed_params:
                    gclid = parsed_params["gclid"][0]
            except Exception:
                pass
        
        # 方法2: 从 custom_fields 提取
        if not gclid:
            custom_fields_key = "custom_fields[gclid]"
            if custom_fields_key in data:
                gclid = data[custom_fields_key]
        
        # 方法3: 直接在顶层查找
        if not gclid and "gclid" in data:
            gclid = data["gclid"]
        
        # ════════════════════════════════════════
        # 日志输出
        # ════════════════════════════════════════
        print("")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║              GUMROAD WEBHOOK RECEIVED                     ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"  Sale ID:      {sale_id}")
        print(f"  Email:        {email}")
        print(f"  Product:      {product_name} ({product_id})")
        print(f"  Amount:       ${amount_usd:.2f}")
        print(f"  GCLID:        {gclid or '❌ NOT FOUND'}")
        print(f"  URL Params:   {url_params}")
        print(f"  Is Test:      {is_test}")
        print(f"  Time:         {datetime.now(timezone.utc).isoformat()}")
        print("")
        
        # 调试：打印所有收到的字段（帮助排查 gclid 位置）
        print("  📦 All fields received:")
        for key, value in sorted(data.items()):
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"     {key}: {val_str}")
        print("")
        
        # ════════════════════════════════════════
        # 上传离线转化到 Google Ads
        # ════════════════════════════════════════
        if gclid and is_test != "true":
            # 格式化时间（Google Ads 需要: "yyyy-mm-dd hh:mm:ss+|-hh:mm"）
            now_utc = datetime.now(timezone.utc)
            sale_time = now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00")
            
            success = await upload_offline_conversion(
                gclid=gclid,
                sale_id=sale_id,
                amount=amount_usd,
                sale_time=sale_time
            )
            
            if success:
                print(f"🎉 转化已上传: ${amount_usd:.2f} from {email}")
            else:
                print(f"⚠️ 转化上传失败，但 webhook 已记录")
        
        elif is_test == "true":
            print("🧪 测试 Ping，跳过 Google Ads 上传")
        
        elif not gclid:
            print("ℹ️ 无 GCLID（非广告流量购买），跳过 Google Ads 上传")
        
        # Gumroad 需要 200 响应，否则会重试
        return Response(status_code=200, content="OK")
    
    except Exception as e:
        print(f"❌ Gumroad webhook 处理异常: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错也返回 200，避免 Gumroad 无限重试
        return Response(status_code=200, content="OK")


@app.get("/api/webhook-test")
def webhook_test():
    """测试 webhook 端点是否可达"""
    google_ads_configured = all([
        os.environ.get("GOOGLE_ADS_CLIENT_ID"),
        os.environ.get("GOOGLE_ADS_CLIENT_SECRET"),
        os.environ.get("GOOGLE_ADS_REFRESH_TOKEN"),
        os.environ.get("GOOGLE_ADS_CUSTOMER_ID"),
        os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN"),
        os.environ.get("GOOGLE_ADS_CONVERSION_ACTION_ID"),
    ])
    
    return {
        "status": "webhook endpoint ready",
        "google_ads_configured": google_ads_configured,
        "missing_env_vars": [
            var for var in [
                "GOOGLE_ADS_CLIENT_ID",
                "GOOGLE_ADS_CLIENT_SECRET", 
                "GOOGLE_ADS_REFRESH_TOKEN",
                "GOOGLE_ADS_CUSTOMER_ID",
                "GOOGLE_ADS_DEVELOPER_TOKEN",
                "GOOGLE_ADS_CONVERSION_ACTION_ID"
            ] if not os.environ.get(var)
        ]
    }


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
