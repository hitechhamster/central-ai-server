# database.py
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DATABASE_NAME = "conversations.db"

@contextmanager
def get_db():
    """数据库连接上下文管理器"""
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_database():
    """初始化数据库表"""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT,
                response_time REAL
            )
        """)
        
        # 创建索引加速查询
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ip 
            ON conversations(ip_address)
        """)
        print("✅ 数据库表已创建")

def save_conversation(ip: str, prompt: str, response: str, model: str = None, response_time: float = None):
    """保存对话记录"""
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO conversations 
                (timestamp, ip_address, prompt, response, model, response_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                ip,
                prompt,
                response,
                model,
                response_time
            ))
        print(f"✅ 对话已保存 - IP: {ip}")
    except Exception as e:
        print(f"❌ 保存对话失败: {e}")

def get_recent_conversations(limit: int = 100):
    """获取最近的对话记录"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, ip_address, prompt, response, model, response_time
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_stats():
    """获取统计信息"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_conversations,
                COUNT(DISTINCT ip_address) as unique_users,
                AVG(response_time) as avg_response_time,
                MIN(timestamp) as first_conversation,
                MAX(timestamp) as last_conversation
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
