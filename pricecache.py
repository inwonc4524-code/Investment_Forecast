"""가격 저장소 — 매번 상장 이후 전체 가격을 다시 받지 않도록 받아 둔 가격을 저장하고,
다음 실행 때는 최근 1달 치만 받아서 이어 붙여요 (빠르고, 야후에 덜 막혀요).

- 이어 붙일 때: 최근 1달 가격과 저장된 가격이 겹치는 첫날의 비율로 옛날 가격을 맞춰요.
  (배당·액면분할이 생기면 야후가 옛날 가격 전체를 다시 계산하는데, 그 비율만큼 옛날 가격을 고치면 같아져요)
- 7일마다 한 번은 전체를 처음부터 다시 받아요 (작은 오차가 쌓이지 않게).
- 저장 파일은 GitHub Actions 캐시에 보관해요 (저장소에는 안 올라가요).
"""
import gzip
import os
import pickle
import time
from datetime import datetime, timezone

import pandas as pd

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pricecache", "prices.pkl.gz")
FULL_EVERY_DAYS = 7
COLS = ["Close", "Adj Close"]
MODE = {"mode": "", "full": 0, "inc": 0, "fallback": 0}


class Raw(dict):
    """yf.download 결과처럼 raw[종목] 으로 꺼내 쓸 수 있는 묶음"""
    columns = pd.MultiIndex.from_tuples([("_", "_")])


def _load():
    try:
        with gzip.open(PATH, "rb") as f:
            obj = pickle.load(f)
        return obj.get("data", {}), obj.get("full_at")
    except Exception:
        return {}, None


def _save(data, full_at):
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        tmp = PATH + ".tmp"
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump({"data": data, "full_at": full_at}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, PATH)
    except Exception as e:
        print("가격 저장 실패:", e)


def _frames(raw, tickers):
    out = {}
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d[[c for c in COLS if c in d.columns]].dropna(how="all")
            if "Close" in d and len(d["Close"].dropna()):
                out[t] = d.copy()
        except Exception:
            pass
    return out


def _grab(tickers, period, batch, pause):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), batch):
        part = tickers[i:i + batch]
        try:
            raw = yf.download(part, period=period, auto_adjust=False,
                              progress=False, group_by="ticker", threads=True)
            out.update(_frames(raw, part))
        except Exception as e:
            print(f"가격 받기 실패({period}):", e)
        if i + batch < len(tickers):
            time.sleep(pause)
    return out


def _splice(old, new):
    """저장된 옛날 가격 + 새로 받은 최근 가격. 겹치는 날이 없거나 이상하면 None"""
    new = new.dropna(subset=["Close"])
    new = new[new["Close"] > 0]
    if old is None or old.empty or new.empty:
        return None
    both = new.index.intersection(old.index)
    if len(both) == 0:
        return None
    d0 = both[0]
    keep = old[old.index < d0].copy()
    for c in COLS:
        if c not in new or c not in old:
            continue
        a, b = new.at[d0, c], old.at[d0, c]
        if not (pd.notna(a) and pd.notna(b) and a > 0 and b > 0):
            return None
        keep[c] = keep[c] * (a / b)
    return pd.concat([keep, new[new.index >= d0]])


def save_state(name, obj):
    """실행 사이에 이어 쓰는 작은 기록(시간별 가격 등)을 가격 저장소 옆에 보관"""
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with gzip.open(os.path.join(os.path.dirname(PATH), f"{name}.pkl.gz"), "wb", compresslevel=3) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"{name} 저장 실패:", e)


def load_state(name, default=None):
    try:
        with gzip.open(os.path.join(os.path.dirname(PATH), f"{name}.pkl.gz"), "rb") as f:
            return pickle.load(f)
    except Exception:
        return default


def download(tickers, batch=120, pause=2, recent="1mo", fresh=None, _nofull=False):
    """종목들의 날마다 가격(Close·Adj Close)을 돌려줌. raw[종목] 으로 꺼내 씀.
    fresh 를 주면 그 종목들만 새로 받고, 나머지는 저장해 둔 가격을 그대로 써요 (장 마감 알림용, 빠르게)."""
    tickers = list(dict.fromkeys(tickers))
    if not _nofull:
        MODE.update(mode="", full=0, inc=0, fallback=0)
    data, full_at = _load()
    if fresh is not None and data:
        fresh = set(fresh)
        keep = {t: data[t] for t in tickers if t in data and t not in fresh}
        need = [t for t in tickers if t not in keep]
        sub = download(need, batch, pause, recent, _nofull=True) if need else Raw()
        MODE["mode"] = "part"
        print(f"이번엔 {len(need)}곳만 새로 받고 {len(keep)}곳은 저장해 둔 가격을 썼어요")
        return Raw({**keep, **sub})
    now = datetime.now(timezone.utc)
    age = (now - full_at).days if full_at else 999
    full = (not data) or (not _nofull and (age >= FULL_EVERY_DAYS or os.environ.get("PRICE_FULL")))
    if full:
        MODE.update(mode="full")
        got = _grab(tickers, "max", batch, pause)
        MODE["full"] = len(got)
        data = {**{t: data[t] for t in tickers if t in data}, **got}
        full_at = now
    else:
        MODE.update(mode="inc")
        have = [t for t in tickers if t in data]
        miss = [t for t in tickers if t not in data]
        rec = _grab(have, recent, max(batch, 200), 1)
        redo = list(miss)
        for t in have:
            s = _splice(data[t], rec[t]) if t in rec else None
            if s is None:
                redo.append(t)
            else:
                data[t] = s
                MODE["inc"] += 1
        if redo:
            got = _grab(redo, "max", batch, pause)
            MODE["fallback"] = len(got)
            data.update(got)
    _save(data, full_at)
    print(f"가격 받기: {'전체 새로' if full else '최근 1달만 이어 붙임'}"
          f" (이어 붙임 {MODE['inc']} · 전체로 받음 {MODE['full'] + MODE['fallback']})"
          + ("" if full else f" · 마지막 전체 받기 {age}일 전"))
    return Raw({t: data[t] for t in tickers if t in data})
