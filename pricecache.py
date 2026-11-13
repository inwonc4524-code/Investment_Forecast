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
KEPT = []   # 야후가 빼먹은 최신 날을 저장해 둔 가격으로 지킨 종목 수 (로그용)
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
    out = pd.concat([keep, new[new.index >= d0]])
    # 야후가 밤사이 마지막 하루(특히 중국·홍콩·싱가포르 ETF)를 잠깐 빼고 줄 때가 있어요.
    # 저장해 둔 가격에 그보다 최신 날(장 마감 알림 때 받은 확정 가격)이 있으면 버리지 않고 이어 붙여요.
    last = new.index[-1]
    tail = old[old.index > last].dropna(subset=["Close"]).tail(3)
    if len(tail) and last in old.index:
        tail = tail.copy()
        for c in COLS:
            if c in tail and c in new and c in old:
                a, b = new.at[last, c], old.at[last, c]
                if pd.notna(a) and pd.notna(b) and a > 0 and b > 0:
                    tail[c] = tail[c] * (a / b)
        out = pd.concat([out, tail])
        KEPT.append(len(tail))
    return out


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
        KEPT.clear()
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
          + ("" if full else f" · 마지막 전체 받기 {age}일 전")
          + (f" · 야후가 빼먹은 최신 날을 저장해 둔 가격으로 지킨 곳 {len(KEPT)}" if KEPT else ""))
    return Raw({t: data[t] for t in tickers if t in data})


# ── 액면분할·자료 오류로 하루 만에 가격이 몇 배씩 튀는 것 바로잡기 ──
# 야후가 일본 ETF 등에서 분할(예: 1주→200주)을 옛날 가격에 반영하지 않거나,
# 분할 전후 하루 이틀만 엉뚱한 값을 줄 때가 있어요. 그러면 '제일 비쌀 때보다 −99%' 같은 숫자가 나와요.
SPLIT_FACTORS = [1, 2, 3, 4, 5, 8, 10, 20, 25, 50, 100, 200, 500, 1000]


def _nice(r, tol=0.06):
    """비율 r 이 분할 비율(1, 1/2, 1/10 …, 2, 10 …)에 가까우면 그 비율, 아니면 None"""
    for f in SPLIT_FACTORS:
        for v in (f, 1 / f):
            if abs(r / v - 1) <= tol:
                return v
    return None


def fix_jumps(s, limit=3.0, name=""):
    """하루 사이 limit 배 넘게 오르거나 1/limit 아래로 떨어진 곳을 찾아,
    앞뒤 가격 비율이 분할 비율과 맞으면 옛날 가격을 그 비율로 맞추고 이상한 날은 지워요.
    분할 비율과 안 맞으면(진짜 폭등·폭락일 수도 있어서) 건드리지 않아요."""
    s = s.dropna()
    s = s[s > 0]
    if len(s) < 3:
        return s
    r = s / s.shift(1)
    bad = [i for i in range(1, len(s)) if r.iloc[i] > limit or r.iloc[i] < 1 / limit]
    if not bad:
        return s
    groups, cur = [], [bad[0]]
    for i in bad[1:]:
        if i - cur[-1] <= 3:      # 사흘 안에 이어진 튐은 한 덩어리
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    vals = s.copy()
    drop = []
    for g in reversed(groups):    # 최근 것부터 고쳐야 옛날 쪽 배율이 겹쳐도 맞아요
        a, b = g[0], g[-1]
        pre, post = vals.iloc[a - 1], vals.iloc[b]
        f = _nice(post / pre)
        if f is None:
            print(f"  {name} {s.index[a].date()} 가격 급변 {post / pre:.3f}배 — 분할 비율이 아니라 그대로 둠")
            continue
        if f != 1:
            vals.iloc[:a] = vals.iloc[:a] * f
        drop += list(s.index[a:b])          # 덩어리 안쪽의 엉뚱한 날 (마지막 날은 정상 가격)
        print(f"  {name} {s.index[a].date()}~{s.index[b].date()} 가격 튐 바로잡음 (옛날 가격 ×{f:g})")
    return vals.drop(index=drop)
