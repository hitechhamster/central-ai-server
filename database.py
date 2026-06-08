# database.py  —  精简存储版（防止磁盘再次撑爆）
import sqlite3
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

# ✅ 自动检测持久化磁盘路径
if os.path.exists("/data"):
    DATABASE_NAME = "/data/conversations.db"
    print("✅ 使用持久化磁盘: /data/conversations.db")
else:
    DATABASE_NAME = "conversations.db"
    print("⚠️ 使用临时存储: conversations.db")

# ---- 存储上限：不再把几千字的 prompt / 六千字的 response 原样塞进库 ----
PROMPT_STORE_CHARS = 200        # prompt 只留头部，够调试即可
RESPONSE_STORE_CHARS = 2000     # response 只留前 2000 字
RETENTION_DAYS_DEFAULT = 30     # /admin/cleanup 默认只保留最近 30 天

# 已知的失败响应（call_llm 在出错时返回的字符串）——用来给每条记录打 status
ERROR_RESPONSES = {
    "服务器配置错误，请联系管理员",
    "请求超时，请稍后重试",
    "AI 服务暂时不可用，请稍后重试",
    "服务出现问题，请稍后重试",
}


@contextmanager
def get_db():
    """数据库连接上下文管理器"""
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _classify_kind(prompt: str) -> str:
    """根据 prompt 内容标记类型（写入时一次性算好，省得以后再 LIKE 搜大字段）"""
    if "MACHINE-READABLE VERDICT" in prompt:
        return "report"   # 主报告 ≈ 一个真实用户看到的解读
    if "STRICT JSON ONLY" in prompt:
        return "email"    # 4 封营销邮件序列的生成调用
    return "other"


def init_database():
    """初始化数据库表（含老库平滑升级）"""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                kind TEXT,
                status TEXT,
                prompt TEXT,
                response TEXT,
                model TEXT,
                response_time REAL
            )
        """)
        # 老库缺列就补上（不影响已部署的旧数据）
        existing = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        if "kind" not in existing:
            conn.execute("ALTER TABLE conversations ADD COLUMN kind TEXT")
        if "status" not in existing:
            conn.execute("ALTER TABLE conversations ADD COLUMN status TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ip ON conversations(ip_address)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kind ON conversations(kind)")
        print("✅ 数据库表已就绪（精简存储版）")


def save_conversation(ip: str, prompt: str, response: str, model: str = None, response_time: float = None):
    """保存对话记录（prompt/response 截断存储 + 自动标记 kind / status）"""
    try:
        prompt = prompt or ""
        response = response or ""
        kind = _classify_kind(prompt)
        status = "failed" if response.strip() in ERROR_RESPONSES else "ok"
        with get_db() as conn:
            conn.execute("""
                INSERT INTO conversations
                (timestamp, ip_address, kind, status, prompt, response, model, response_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                ip,
                kind,
                status,
                prompt[:PROMPT_STORE_CHARS],
                response[:RESPONSE_STORE_CHARS],
                model,
                response_time,
            ))
    except Exception as e:
        print(f"❌ 保存对话失败: {e}")


def cleanup(days: int = RETENTION_DAYS_DEFAULT, vacuum: bool = True):
    """删除超过 N 天的记录，并（可选）VACUUM 回收磁盘空间。

    注意：VACUUM 需要约等于库大小的临时空间，磁盘满时跑不动，
    必须先腾出空间（删 .db 文件或扩容）后再调用。
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with get_db() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount

    vacuumed = False
    if vacuum:
        # VACUUM 不能在事务中执行，单开一个连接
        conn2 = sqlite3.connect(DATABASE_NAME)
        try:
            conn2.execute("VACUUM")
            conn2.commit()
            vacuumed = True
        except Exception as e:
            print(f"⚠️ VACUUM 失败（可能磁盘空间不足）: {e}")
        finally:
            conn2.close()

    print(f"🧹 清理完成：删除 {deleted} 行，保留最近 {days} 天 | VACUUM={vacuumed}")
    return {"deleted": deleted, "retention_days": days, "vacuumed": vacuumed}


def get_recent_conversations(limit: int = 100):
    """获取最近的对话记录"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, ip_address, kind, status, prompt, response, model, response_time
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_stats():
    """获取统计信息（含真实用户数 / 失败数）"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN kind='report' THEN 1 ELSE 0 END) AS total_reports,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS total_failed,
                COUNT(DISTINCT ip_address) AS unique_ips,
                AVG(response_time) AS avg_response_time,
                MIN(timestamp) AS first_conversation,
                MAX(timestamp) AS last_conversation
            FROM conversations
        """)
        return dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))


def search_conversations(keyword: str = None, ip: str = None, limit: int = 50):
    """搜索对话记录"""
    with get_db() as conn:
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        if keyword:
            query += " AND (prompt LIKE ? OR response LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        if ip:
            query += " AND ip_address = ?"
            params.append(ip)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
