import re
import hashlib
from typing import List, Dict, Optional, Any, TypeAlias
from urllib.parse import urlparse
from datetime import datetime, timezone

import numpy as np

TavilySearchResult: TypeAlias = Dict[str, Any]
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]

def clip_to_token_budget(text: str, max_tokens: int) -> str:
    approx_chars = max_tokens * 4
    return text if len(text) <= approx_chars else text[:approx_chars]

def canonicalize(url: str) -> str:
    p = urlparse(url or "")
    scheme = p.scheme or "https"
    netloc = (p.netloc or "").lower()
    path = re.sub(r"/+$", "", p.path or "")
    return f"{scheme}://{netloc}{path}"

def domain(url: str) -> str:
    return (urlparse(url or "").netloc or "").lower()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

_VENDOR_PAT = re.compile(r"(ads|sponsored|utm_|/press/|/blog/|/solutions/)", re.I)
PRIMARY_AUTHORITIES = {
    "ieee.org","rfc-editor.org","w3.org","who.int","nasa.gov","nature.com","acm.org","arxiv.org",
    "nist.gov","iso.org","owasp.org","docs.python.org","nih.gov","oecd.org",".ac.uk"
}

def freshness_score(pub_date: Optional[str], event_date: Optional[str]=None) -> float:
    ref = pub_date or event_date
    if not ref:
        return 0.3
    try:
        pub = datetime.fromisoformat(ref)
    except Exception:
        return 0.3
    delta_days = (datetime.now(timezone.utc) - pub).days
    if delta_days <= 90: return 1.0
    if delta_days <= 365: return 0.7
    if delta_days <= 3*365: return 0.4
    return 0.2

def is_vendor(url: str) -> bool:
    d = domain(url)
    return (d.endswith(".com") or d.endswith(".io") or d.endswith(".ai")) and not any(a in d for a in PRIMARY_AUTHORITIES)

def bias_level(url: str) -> str:
    return "med" if is_vendor(url) else ("low" if _VENDOR_PAT.search(url or "") else "none")

def authority_weight(url: str, pub_date: Optional[str]=None, has_methods: Optional[bool]=None) -> float:
    d = domain(url)
    base = 0.2
    if any(d.endswith(suf) for suf in (".gov",".edu",".ac.uk",".int")): base = 0.8
    if any(auth in d for auth in PRIMARY_AUTHORITIES): base = max(base, 0.8)
    if d.endswith(".org"): base = max(base, 0.6)
    if is_vendor(url): base = min(base, 0.35)
    rec = freshness_score(pub_date, None)
    base += (0.10 if rec >= 0.7 else -0.15 if rec <= 0.2 else 0.0)
    if has_methods is True: base += 0.05
    if has_methods is False: base -= 0.10
    return float(max(0.05, min(1.0, base)))

def hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
