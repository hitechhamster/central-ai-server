"""
邮箱可投递性校验 —— 只用在**抓线索**这一步,不碰交付链路。

为什么要有它:免费八字计算器强制留邮箱,用户想跳过就乱敲。实测(2026-08-10 Klaviyo)
**约 18% 的新订阅是无效地址**,退信里 49% 是 Invalid Address、50% 是硬退。
这些假地址每个都会先收到一封信才被发现,持续消耗发信信誉。

三层,越往后越贵:
  ① 格式        —— 微秒级
  ② 垃圾域名名单 —— 微秒级,拦 fjif.fj / je.dj 这类键盘乱敲
  ③ MX 记录     —— 一次 DNS,拦 hotmdne.com / gmail.fr / gjid.co 这种"像真的"假域名

⚠️ **一律 fail-open**:DNS 超时、解析异常、库没装 —— 任何不确定都放行。
   宁可漏掉几个假邮箱,也绝不能因为 DNS 抽风把真实用户挡在门外。

⚠️ MX 查询对**假域名最慢**(要等不存在的域名超时),所以超时钉死 1.5 秒。
   加上域名级缓存,gmail/yahoo/qq 这些占绝大多数流量的域名是 0 毫秒。
"""
import re
import time

# 常见一次性/垃圾邮箱域名。不求全,拦掉高频的即可 —— 真正的主力是 MX 检查。
DISPOSABLE = {
    'mailinator.com', 'guerrillamail.com', 'guerrillamail.net', '10minutemail.com',
    'tempmail.com', 'temp-mail.org', 'throwawaymail.com', 'yopmail.com', 'trashmail.com',
    'sharklasers.com', 'getnada.com', 'maildrop.cc', 'dispostable.com', 'fakeinbox.com',
    'mailnesia.com', 'mytemp.email', 'moakt.com', 'emailondeck.com', 'spamgourmet.com',
}

# 常见拼错。这些域名可能真实存在(被抢注),但收件人几乎必然是笔误,发过去就是硬退。
TYPO_DOMAINS = {
    'gmail.co', 'gmail.cm', 'gmail.con', 'gmail.fr', 'gmai.com', 'gmial.com', 'gmail.om',
    'yahoo.co', 'yaho.com', 'hotmial.com', 'hotmai.com', 'outlok.com', 'iclod.com',
}

_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@([A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
                 r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)+)$")

MX_TIMEOUT = 1.5          # 秒。假域名要等满这个时间,所以不能设大
CACHE_TTL = 24 * 3600     # 域名级缓存,一天
CACHE_MAX = 5000

_cache: dict[str, tuple[bool, float]] = {}


def _cache_get(domain):
    hit = _cache.get(domain)
    if not hit:
        return None
    ok, ts = hit
    if time.time() - ts > CACHE_TTL:
        _cache.pop(domain, None)
        return None
    return ok


def _cache_put(domain, ok):
    if len(_cache) >= CACHE_MAX:            # 简单粗暴地清一半,不值得上 LRU
        for k in list(_cache)[: CACHE_MAX // 2]:
            _cache.pop(k, None)
    _cache[domain] = (ok, time.time())


async def _has_mx(domain: str) -> bool:
    """
    域名能不能收信。**任何异常都返回 True(放行)** —— 见模块开头的 fail-open 原则。
    用 dnspython 的异步 resolver,避免阻塞事件循环。
    """
    cached = _cache_get(domain)
    if cached is not None:
        return cached
    try:
        import dns.asyncresolver
        import dns.resolver
    except ImportError:
        return True                          # 库没装 → 这一层直接跳过,不报错
    try:
        r = dns.asyncresolver.Resolver()
        r.timeout = MX_TIMEOUT
        r.lifetime = MX_TIMEOUT
        try:
            ans = await r.resolve(domain, 'MX')
            ok = len(ans) > 0
        except dns.resolver.NoAnswer:
            # 没有 MX 但有 A 记录时,按 RFC 5321 仍可投递到该主机
            try:
                a = await r.resolve(domain, 'A')
                ok = len(a) > 0
            except Exception:
                ok = False
        except (dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
            ok = False                       # 域名根本不存在 —— 这是我们要抓的主要目标
    except Exception:
        return True                          # 超时/网络问题 → 放行,不缓存
    _cache_put(domain, ok)
    return ok


async def check_email(email: str) -> tuple[bool, str]:
    """
    返回 (是否放行, 原因)。原因只用于服务端日志,别原样吐给用户 ——
    对着用户说"你的域名没有 MX 记录"没有任何意义。
    """
    e = (email or '').strip()
    m = _RE.match(e)
    if not m:
        return False, 'bad_format'
    domain = m.group(1).lower()
    if domain in DISPOSABLE:
        return False, 'disposable'
    if domain in TYPO_DOMAINS:
        return False, 'typo_domain'
    if not await _has_mx(domain):
        return False, 'no_mx'
    return True, 'ok'
