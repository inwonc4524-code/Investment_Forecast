"""
ETF 추천 봇
- etfs.csv 에 적힌 한국·미국·중국·홍콩 ETF의 가격을 받아와서
  ① 요즘 추천 (쑥쑥 오르는 중 / 쉬는 중 / 천천히 꾸준히)
  ② 오래 꾸준히 오른 ETF (1·3·5·10·20년)
  를 계산하고, 휴대폰(ntfy)으로 요약을 보내고, docs/index.html 대시보드를 만듭니다.

실행:
  python etf_bot.py          # 실제 데이터 (GitHub Actions에서 자동 실행)
  python etf_bot.py --demo   # 가짜 데이터로 동작 확인 (인터넷 없이)

※ 이 봇의 결과는 과거 가격으로 계산한 참고 자료일 뿐, 투자 권유가 아닙니다.
"""

import argparse
import html
import json
import re
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

KST = timezone(timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
TOP_N = 10                   # 추천 카드마다 보여줄 개수
LONG_SHOW = 50               # 순위 목록에 넣을 개수 (11위부터는 더보기)
FIRST_SHOW = 10              # 처음에 보여줄 개수
MIN_DAYS = 220               # 이보다 데이터가 짧으면(상장 1년 미만 등) 제외
PERIODS = ["1d", "1w", "1m", 1, 3, 5, 10, 20]  # '오래 꾸준히' 기간
PERIOD_INFO = {   # 기간: (버튼 이름, 칸 하나의 단위, 세는 말, 기간 이름)
    "1d": ("1일", "시간", "시간", "1일"), "1w": ("1주", "날", "일", "1주"), "1m": ("1달", "주", "주", "1달"),
    1: ("1년", "달", "달", "1년"), 3: ("3년", "해", "번", "3년"), 5: ("5년", "해", "번", "5년"),
    10: ("10년", "해", "번", "10년"), 20: ("20년", "해", "번", "20년"),
}

MARKETS = [("KR", "🇰🇷", "한국"), ("US", "🇺🇸", "미국"),
           ("CN", "🇨🇳", "중국"), ("HK", "🇭🇰", "홍콩")]
FLAG = {c: f for c, f, _ in MARKETS}

THEMES = [   # 선택해서 보는 테마 탭 — etfs.csv 의 theme 칸에 key를 적어요 (여러 개면 | 로)
    ("sp500", "🇺🇸 미국 S&P500", "미국 대표 회사 500개를 담은 ETF예요"),
    ("tech", "💻 기술·나스닥", "애플·마이크로소프트·엔비디아 같은 기술 회사를 담은 ETF예요"),
    ("semi", "💾 반도체", "반도체 회사들을 담은 ETF예요"),
    ("bat", "🔋 2차전지", "배터리·전기차 회사들을 담은 ETF예요"),
    ("ess", "⚡ 전고체·ESS·친환경", "차세대 배터리·전기 저장·태양광 같은 친환경 에너지 ETF예요"),
    ("power", "🏭 AI 전력·원자력", "데이터센터 전기, 전력망, 원자력 회사를 담은 ETF예요"),
    ("robot", "🤖 로봇·AI", "로봇과 인공지능 회사를 담은 ETF예요"),
    ("bio", "🧬 바이오·헬스케어", "약·병원·의료기기 회사를 담은 ETF예요"),
    ("defense", "🛡 방산·조선", "무기·배를 만드는 회사를 담은 ETF예요"),
    ("div", "💰 배당", "회사가 나눠주는 돈(배당)을 많이 주는 회사들을 담은 ETF예요"),
    ("gold", "🥇 금·원자재", "금·은·석유 같은 물건 값을 따라가는 ETF예요"),
    ("bond", "🧾 채권", "나라나 회사에 돈을 빌려주고 이자를 받는 ETF예요. 덜 출렁여요"),
    ("world", "🌏 해외 나라별", "일본·인도·대만 같은 다른 나라 시장을 담은 ETF예요"),
]
THEME_NAME = {k: n for k, n, _ in THEMES}


def in_theme(df, key):
    if key is None:
        return df
    return df[df.theme.fillna("").str.split("|").apply(lambda ts: key in ts)]


# ───────────────────────── 1. 데이터 받아오기 ─────────────────────────

CLOSE_KST = {"KR": "15:30", "US": "06:00", "CN": "16:00", "HK": "17:00"}   # 각 나라 장 마감 (한국시간)
MKT_NAME = {"KR": "한국", "US": "미국", "CN": "중국", "HK": "홍콩"}


def load_universe():
    u = pd.read_csv(os.path.join(HERE, "etfs.csv"), dtype=str)
    if "theme" not in u:
        u["theme"] = ""
    u["theme"] = u.theme.fillna("").str.replace(" ", "")
    return u


RAW = {}   # 실제 가격(배당 조정 안 한 종가) — '실제 가격으로 얼마 올랐나' 표시에 사용


def _pick(raw, key, name):
    """다운로드 결과에서 수익률 계산용(배당 포함 수정종가)을 돌려주고, 실제 가격은 RAW에 저장"""
    d = raw[key] if isinstance(raw.columns, pd.MultiIndex) else raw
    real = d["Close"].dropna()
    real = real[real > 0]
    adj = (d["Adj Close"] if "Adj Close" in d else d["Close"]).dropna()
    adj = adj[adj > 0]
    if len(real):
        RAW[name] = real
    return adj


INTRA = {}   # 1일: 마지막 장날의 1시간 단위 가격 {"date", "open", "close": [...], "times": [...]}


MIN_HOURS = 3   # 1일 화면: 1시간 칸이 이보다 적게 오면 '하루 전체' 한 칸으로 보여줌
TZ_BY_SUFFIX = {".KS": "Asia/Seoul", ".KQ": "Asia/Seoul", ".SS": "Asia/Shanghai",
                ".SZ": "Asia/Shanghai", ".HK": "Asia/Hong_Kong", ".T": "Asia/Tokyo",
                ".TW": "Asia/Taipei", ".TWO": "Asia/Taipei", ".SI": "Asia/Singapore"}


CLOSE_LOCAL = {"Asia/Seoul": "15:30", "Asia/Shanghai": "15:00", "Asia/Hong_Kong": "16:00",
               "Asia/Tokyo": "15:30", "Asia/Taipei": "13:30", "Asia/Singapore": "17:00",
               "America/New_York": "16:00"}   # 정규장 마감 (그 나라 시간)


def fill_close(t, it):
    """야후가 마지막 30분 등을 빼고 줄 때가 있어서, 그날 실제 마감 가격으로 마지막 칸을 채움"""
    real = RAW.get(t)
    if real is None or real.empty:
        return
    day = it["date"].normalize()
    same = real[real.index.normalize() == day]
    if same.empty:
        return
    final = float(same.iloc[-1])
    last = it["times"][-1]
    hh, mm = CLOSE_LOCAL[local_tz(t)].split(":")
    close_t = last.normalize() + pd.Timedelta(hours=int(hh), minutes=int(mm))
    if last + pd.Timedelta(hours=1) < close_t:      # 마감 전에 빈 시간이 있음 → 칸 추가
        it["close"].append(final)
        it["times"].append(last + pd.Timedelta(hours=1))
        it["tail"] = True
    else:                                           # 마지막 칸이 마감까지 → 마감 가격으로 맞춤
        it["close"][-1] = final


def local_tz(ticker):
    """종목코드 끝(.KS 등)을 보고 그 나라 시간대를 정함. 끝이 없으면 미국."""
    for suf, tz in TZ_BY_SUFFIX.items():
        if ticker.endswith(suf):
            return tz
    return "America/New_York"


def fetch_intraday(tickers):
    """마지막 장날, 장 시작부터 마감까지 1시간마다 가격"""
    import yfinance as yf
    try:
        raw = yf.download(tickers, period="5d", interval="60m", auto_adjust=False,
                          progress=False, group_by="ticker", threads=True, prepost=False)
    except Exception as e:
        print("시간별 가격을 못 받았어요:", e)
        return
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d.dropna(subset=["Open", "Close"])
            d = d[d.Close > 0]
            if d.empty:
                continue
            idx = d.index if d.index.tz is not None else d.index.tz_localize("UTC")
            d = d.set_axis(idx.tz_convert(local_tz(t)))
            # 야후가 1시간 가격 대신 하루치 가격 1개(자정 00:00)만 주는 경우가 있어서 걸러냄
            d = d[[not (x.hour == 0 and x.minute == 0) for x in d.index]]
            if d.empty:
                continue
            days = [x.date() for x in d.index]
            last = days[-1]
            day = d[[x == last for x in days]]
            if len(day) < MIN_HOURS:      # 1시간 칸이 너무 적으면 믿지 않고 '하루 전체'로 보여줌
                continue
            INTRA[t] = {"date": pd.Timestamp(last), "open": float(day.Open.iloc[0]),
                        "close": [float(x) for x in day.Close], "times": list(day.index)}
        except Exception:
            pass
    for t, it in INTRA.items():
        try:
            fill_close(t, it)
        except Exception:
            pass
    print(f"시간별 가격을 받은 곳: {len(INTRA)}/{len(tickers)}")


def demo_intraday(closes, seed=3):
    rng = np.random.default_rng(seed)
    for t, s in closes.items():
        o = float(s.iloc[-2])
        moves = rng.normal(0, 0.006, 7)
        cl = list(o * np.exp(np.cumsum(moves)))
        start = pd.Timestamp(s.index[-1]) + pd.Timedelta(hours=9)
        INTRA[t] = {"date": pd.Timestamp(s.index[-1]), "open": o, "close": cl,
                    "times": [start + pd.Timedelta(hours=h) for h in range(7)]}


def apply_intraday(df):
    """1일 = 그 나라 장 시작 → 장 마감 (시간별 가격이 없으면 전날 마감 대비)"""
    def one(r):
        it = INTRA.get(r.ticker)
        if it and it["date"].strftime("%Y-%m-%d") == r.date:
            return it["close"][-1] / it["open"] - 1
        return r.r1d
    df["r1d"] = df.apply(one, axis=1)
    return df


def intraday_term(ticker, s=None):
    it = INTRA.get(ticker)
    if not it:
        if s is None or len(s.dropna()) < 2:
            return None
        s = s.dropna()
        real = RAW.get(ticker) if "RAW" in globals() else None
        p0, p1 = float(s.iloc[-2]), float(s.iloc[-1])
        if real is not None and len(real) >= 2:
            p0, p1 = float(real.iloc[-2]), float(real.iloc[-1])
        total = float(s.iloc[-1] / s.iloc[-2] - 1)
        return {"total": total, "cagr": np.nan, "blocks": [total], "labels": ["하루 전체"],
                "ups": int(total > 0), "start": s.index[-1], "end": s.index[-1],
                "p0": p0, "p1": p1, "daily": True, "is1d": True}
    o, cl = it["open"], it["close"]
    prev = [o] + cl[:-1]
    blocks = [c / p - 1 for c, p in zip(cl, prev)]
    close_hm = CLOSE_LOCAL[local_tz(ticker)]
    hh, mm = map(int, close_hm.split(":"))

    def lab(t):
        end = t.normalize() + pd.Timedelta(hours=hh, minutes=mm)
        if t + pd.Timedelta(hours=1) > end:      # 1시간이 안 되는 마지막 칸
            return f"{t:%H:%M}부터 마감까지 ({close_hm})"
        return f"{t:%H:%M}부터 1시간"
    labels = [lab(t) for t in it["times"]]
    return {"total": cl[-1] / o - 1, "cagr": np.nan, "blocks": blocks, "labels": labels,
            "ups": sum(b > 0 for b in blocks), "start": it["date"], "end": it["date"],
            "p0": o, "p1": cl[-1], "is1d": True}


def fetch_prices(tickers):
    """야후 파이낸스에서 상장 이후 전체 기간의 가격(배당 포함 수정종가)을 받아옵니다."""
    import yfinance as yf

    raw = yf.download(tickers, period="max", auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    closes = {}
    for t in tickers:
        try:
            s = _pick(raw, t, t)
            s = s.dropna()
            s = s[s > 0]
            if len(s) > 0:
                closes[t] = s
        except Exception:
            pass
    # 잠깐 오류로 못 받은 종목은 하나씩 한 번 더 받아오기
    retry = [t for t in tickers if t not in closes]
    if retry:
        raw_r = yf.download(retry, period="max", auto_adjust=False,
                            progress=False, group_by="ticker", threads=False)
        for t in retry:
            try:
                s = _pick(raw_r, t, t)
                if len(s) > 0:
                    closes[t] = s
            except Exception:
                pass
    missing = [t for t in tickers if t not in closes]
    if missing:
        print("가격을 못 받은 종목:", ", ".join(missing))
    return closes


def demo_prices(tickers, seed=7):
    """인터넷 없이 테스트할 때 쓰는 가짜 가격 (종목마다 2~22년치)."""
    rng = np.random.default_rng(seed)
    closes = {}
    for t in tickers:
        years = int(rng.integers(2, 23))
        idx = pd.bdate_range(end=datetime.now(KST).date(), periods=252 * years)
        drift = rng.normal(0.0003, 0.0004)
        vol = rng.uniform(0.004, 0.022)
        r = rng.normal(drift, vol, len(idx))
        r[-25:] += rng.normal(0, 0.004)
        closes[t] = pd.Series(10000 * np.exp(np.cumsum(r)), index=idx)
    RAW.update(closes)
    return closes


# ───────────────────────── 2. 요즘 지표 계산 ─────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def ret(s, days):
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1) if len(s) > days else np.nan


def ret_days(s, days):
    """달력으로 며칠 전 마감 가격과 비교 (1주=7일, 1달=4주=28일)"""
    start = s.index[-1] - pd.Timedelta(days=days)
    if s.index[0] > start:
        return np.nan
    return float(s.iloc[-1] / s.asof(start) - 1)


def indicators(s):
    s = s.dropna()
    last_year = s.iloc[-252:]
    ma200 = s.rolling(200).mean()
    daily = s.pct_change().dropna()
    running_max = last_year.cummax()
    return {
        "price": float(s.iloc[-1]),
        "date": s.index[-1].strftime("%Y-%m-%d"),
        "since": s.index[0].strftime("%Y-%m-%d"),
        "r1d": ret(s, 1),
        "r1w": ret_days(s, 7),
        "r1m": ret_days(s, 28),
        "r3m": ret(s, 63),
        "r6m": ret(s, 126),
        "r12m": ret(s, 252),
        "above_ma200": bool(s.iloc[-1] > ma200.iloc[-1]),
        "ma200_rising": bool(ma200.iloc[-1] > ma200.iloc[-21]),
        "rsi14": rsi(s),
        "vol60": float(daily.iloc[-60:].std() * np.sqrt(252)),
        "mdd1y": float((last_year / running_max - 1).min()),
        "from_high": float(s.iloc[-1] / last_year.max() - 1),
    }


def pct_rank(series, ascending=True):
    r = series.rank(pct=True, ascending=ascending)
    return (r * 100).round(1)


def score(df):
    # 쑥쑥: 1·3·6개월 수익률 가중 평균, 200일 평균선 위에 있어야 후보
    df["mom_raw"] = 0.2 * df.r1m + 0.4 * df.r3m + 0.4 * df.r6m
    df["mom_score"] = pct_rank(df.mom_raw)
    df.loc[~df.above_ma200, "mom_score"] *= 0.5

    # 쉬는 중: 긴 흐름(200일 평균선)은 오르는 중인데 요즘 많이 내려온 것
    df["rebound_score"] = (0.5 * pct_rank(df.rsi14, ascending=False)
                           + 0.5 * pct_rank(df.from_high, ascending=False)).round(1)
    df["rebound_ok"] = df.ma200_rising & ((df.rsi14 < 40) | (df.from_high <= -0.08))

    # 천천히 꾸준히: 흔들림·최대 하락폭이 작고 6개월 수익은 플러스
    df["stab_score"] = (0.6 * pct_rank(df.vol60, ascending=False)
                        + 0.4 * pct_rank(df.mdd1y)).round(1)
    df["stab_ok"] = df.r6m > 0

    # 오늘의 추천: 쑥쑥 60% + 안정 40%, 너무 빨리 오른 것(RSI 75↑)은 감점
    df["total"] = (0.6 * df.mom_score + 0.4 * df.stab_score).round(1)
    df.loc[df.rsi14 >= 75, "total"] -= 10
    df["overheated"] = df.rsi14 >= 70
    return df


# ───────────────────────── 3. 오래 꾸준히 오른 ETF ─────────────────────────

def long_term(s, years):
    """기간 동안의 성적. 데이터가 모자라면 None.
    1주·1달: 하루씩, 1년: 12달을 한 달씩, 3년 이상: 1년씩 잘라서 몇 번 올랐는지 셉니다."""
    end = s.index[-1]
    if years == "1w":   # 달력 7일, 하루씩 (장이 쉰 날은 None)
        if len(s) < 8:
            return None
        days = [end - pd.Timedelta(days=k) for k in range(6, -1, -1)]
        blocks, labels = [], []
        for d in days:
            wd = "월화수목금토일"[d.weekday()]
            labels.append(f"{d.month}월 {d.day}일 ({wd})")
            if d in s.index:
                prev = s[s.index < d]
                blocks.append(float(s[d] / prev.iloc[-1] - 1) if len(prev) else None)
            else:
                blocks.append(None)
        start = end - pd.Timedelta(days=7)
        total = float(s.iloc[-1] / s.asof(start) - 1)
        return {"total": total, "cagr": np.nan, "blocks": blocks, "start": start, "end": end,
                "labels": labels, "ups": sum(1 for b in blocks if b is not None and b > 0)}
    if years == "1m":   # 4주, 일주일씩
        start = end - pd.Timedelta(days=28)
        if s.index[0] > start:
            return None
        cuts = [end - pd.Timedelta(days=7 * k) for k in range(4, -1, -1)]
        px = [float(s.asof(c)) for c in cuts]
        blocks = [px[i + 1] / px[i] - 1 for i in range(4)]
        labels = [f"{(a + pd.Timedelta(days=1)).month}/{(a + pd.Timedelta(days=1)).day}~{b.month}/{b.day}"
                  for a, b in zip(cuts[:-1], cuts[1:])]
        return {"total": px[-1] / px[0] - 1, "cagr": np.nan, "blocks": blocks, "start": start,
                "end": end, "labels": labels, "ups": sum(b > 0 for b in blocks)}
    start = end - pd.DateOffset(years=years)
    if s.index[0] > start + pd.Timedelta(days=15):
        return None
    if years == 1:
        cuts = [end - pd.DateOffset(months=m) for m in range(12, -1, -1)]
        labels = [(c + pd.DateOffset(days=1)).strftime("%y년 %m월") for c in cuts[1:]]
    else:
        cuts = [end - pd.DateOffset(years=y) for y in range(years, -1, -1)]
        labels = [f"{c.strftime('%Y.%m')}~{d.strftime('%Y.%m')}" for c, d in zip(cuts[:-1], cuts[1:])]
    px = [float(s.asof(c)) for c in cuts]
    blocks = [px[i + 1] / px[i] - 1 for i in range(len(px) - 1)]
    total = px[-1] / px[0] - 1
    return {
        "total": total,
        "start": cuts[0],
        "cagr": (1 + total) ** (1 / years) - 1,
        "blocks": blocks,
        "labels": labels,
        "ups": sum(b > 0 for b in blocks),
    }


def build_long(universe, closes):
    """{기간: [ETF별 성적 ...]}"""
    out = {}
    for y in PERIODS:
        rows = []
        for _, u in universe.iterrows():
            s = closes.get(u.ticker)
            if s is None or s.empty:
                continue
            lt = intraday_term(u.ticker, s) if y == "1d" else long_term(s.dropna(), y)
            if lt:
                real = RAW.get(u.ticker)
                if y != "1d" and real is not None and len(real):
                    lt["p0"] = float(real.asof(lt["start"])) if real.index[0] <= lt["start"] else np.nan
                    lt["p1"] = float(real.iloc[-1])
                rows.append({**u.to_dict(), **lt})
        out[y] = rows
    return out


def long_rank(rows):
    """올라간 횟수가 많은 순 → 같으면 1년 평균 수익률 높은 순. 전체로 손해 본 건 제외."""
    good = [r for r in rows if r["total"] > 0]
    if good and good[0].get("is1d"):     # 1일: 1시간 칸 수가 종목마다 달라서 그날 많이 오른 순
        return sorted(good, key=lambda r: r["total"], reverse=True)
    return sorted(good, key=lambda r: (r["ups"] / max(1, sum(b is not None for b in r["blocks"])),
                                       r["total"] if pd.isna(r["cagr"]) else r["cagr"]), reverse=True)


# ───────────────────────── 4. 공통 ─────────────────────────

def build_table(universe, closes):
    rows = []
    for _, u in universe.iterrows():
        s = closes.get(u.ticker)
        if s is None or len(s.dropna()) < MIN_DAYS:
            continue
        rows.append({**u.to_dict(), **indicators(s)})
    if not rows:
        raise SystemExit("계산할 수 있는 ETF가 없습니다. 데이터 연결을 확인하세요.")
    return score(pd.DataFrame(rows))


def th(r):
    """화면에서 테마로 거를 때 쓰는 값"""
    return E(str(r.get("theme", "") or ""))


def combo_keep(items, n, mk, th):
    """정렬된 목록에서, (전체/나라) × (전체/테마) 조합마다 상위 n개에 드는 것만 남김"""
    cnt, out = {}, []
    for it in items:
        ms = ("all", mk(it))
        ts = [""] + [t for t in str(th(it) or "").split("|") if t]
        keys = [(m, t) for m in ms for t in ts]
        if any(cnt.get(k, 0) < n for k in keys):
            out.append(it)
        for k in keys:
            cnt[k] = cnt.get(k, 0) + 1
    return out


def picks_all(df):
    """대시보드용: 조합마다 상위 TOP_N개가 다 들어가도록 넉넉하게"""
    full = {
        "overall": df[df.above_ma200].sort_values("total", ascending=False),
        "momentum": df[df.above_ma200 & (df.mom_raw > 0)].sort_values("mom_score", ascending=False),
        "rebound": df[df.rebound_ok].sort_values("rebound_score", ascending=False),
        "stable": df[df.stab_ok].sort_values("stab_score", ascending=False),
    }
    if "value_ok" in df:
        full["value"] = df[df.value_ok].sort_values("fin_score", ascending=False)
    out = {}
    for k, d in full.items():
        rows = combo_keep(list(d.iterrows()), TOP_N, lambda x: x[1].market, lambda x: x[1].get("theme", ""))
        out[k] = d.loc[[i for i, _ in rows]]
    return out


def picks(df):
    return {
        "overall": df[df.above_ma200].nlargest(TOP_N, "total"),
        "momentum": df[df.above_ma200 & (df.mom_raw > 0)].nlargest(TOP_N, "mom_score"),
        "rebound": df[df.rebound_ok].nlargest(TOP_N, "rebound_score"),
        "stable": df[df.stab_ok].nlargest(TOP_N, "stab_score"),
    }


def pct(x):
    return "-" if pd.isna(x) else f"{x * 100:+.0f}%" if abs(x) >= 0.1 else f"{x * 100:+.1f}%"


def shake(v):
    """흔들림(1년 환산 변동성)을 쉬운 말로"""
    if v < 0.12:
        return "작음"
    if v < 0.22:
        return "보통"
    return "큼"


def short_ticker(t):
    return t.split(".")[0]


def money(v, market):
    """나라별 돈 단위로"""
    if pd.isna(v):
        return "-"
    if market == "KR":
        return f"{v:,.0f}원"
    if market == "US":
        return f"${v:,.2f}"
    if market == "CN":
        return f"{v:,.2f}위안"
    if market == "HK":
        return f"HK${v:,.2f}"
    return f"{v:,.2f}"


def load_reasons():
    """reasons.json — 기간별로 '왜 올랐나' 한 줄 (Claude가 뉴스를 보고 정리)"""
    try:
        with open(os.path.join(HERE, "reasons.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


REASONS = load_reasons()


def reason_for(ticker, years, end=None):
    r = REASONS.get(ticker) or {}
    if years == "1d":   # 1일 이유는 그 날짜 것만 (예전 날 이유가 남아 있으면 안 보여줌)
        if end is None or r.get("1d_date") != pd.Timestamp(end).strftime("%Y-%m-%d"):
            return ""
    key = years if years in ("1d", "1w", "1m") else "1" if years == 1 else "long"
    return r.get(key, "")


def won(total):
    """100만 원이 얼마가 됐는지"""
    v = 100 * (1 + total)
    return f"{v:,.0f}만 원" if v < 10000 else f"{v / 10000:.1f}억 원"


# ───────────────────────── 5. 휴대폰 알림 ─────────────────────────

def phone_message(df, long, today):
    out = [f"📅 {today} 기준"]
    for code, flag, label in MARKETS:
        sub = df[df.market == code]
        if sub.empty:
            continue
        p = picks(sub)
        out.append(f"\n{flag} {label}")
        for icon, key, extra in [
            ("🏆 오늘의 추천", "overall", lambda r: f"{r.total:.0f}점"),
            ("🚀 쑥쑥 오르는 중", "momentum", lambda r: f"6달 동안 {pct(r.r6m)}"),
            ("🎢 잠깐 쉬는 중", "rebound", lambda r: f"제일 비쌀 때보다 {pct(r.from_high)}"),
            ("🐢 천천히 꾸준히", "stable", lambda r: f"흔들림 {shake(r.vol60)}"),
        ]:
            rows = p[key]
            if rows.empty:
                out.append(f"{icon}: 없음")
            else:
                r = rows.iloc[0]
                hot = " 🔥" if r.overheated else ""
                out.append(f"{icon}: {r['name']} ({short_ticker(r.ticker)}) {extra(r)}{hot}")
    best = long_rank(long.get(10, []))
    if best:
        b = best[0]
        out.append(f"\n🌱 10년 꾸준히 1등: {FLAG.get(b['market'], '')} {b['name']}"
                   f" — 10번 중 {b['ups']}번 오름, 100만 원 → {won(b['total'])}")
    us = df[(df.market == "US") & (df.r1d > 0)].sort_values("r1d", ascending=False).head(3)
    if not us.empty:
        d = pd.Timestamp(us.date.max())
        names = ", ".join(f"{r['name']} {pct(r.r1d)}" for _, r in us.iterrows())
        out.append(f"\n🇺🇸 미국장 마감({d.month}월 {d.day}일) 하루 동안 많이 오른 곳: {names}")
    out.append("\n🔥 = 너무 빨리 올라서 조심")
    out.append("※ 지나간 숫자로 계산한 참고용이에요. 꼭 오른다는 뜻도, 사라는 뜻도 아니에요.")
    return "\n".join(out)


def close_message(df, market, today):
    """그 나라 장이 끝났을 때: 오늘 하루 많이 오른 종목"""
    sub = df[df.market == market].dropna(subset=["r1d"]).sort_values("r1d", ascending=False)
    flag = FLAG.get(market, "")
    name = MKT_NAME.get(market, market)
    if sub.empty:
        return None
    last = sub.date.max()
    late = "" if last == today else f"\n(아직 오늘 가격이 안 들어와서 {last} 기준이에요)"
    up = sub[sub.r1d > 0]
    out = [f"{flag} {name} 주식시장이 끝났어요 ({CLOSE_KST.get(market, '')} 마감){late}", ""]
    if up.empty:
        out.append("오늘은 오른 곳이 없어요. 다 같이 쉬는 날이에요 🛌")
    else:
        out.append("🔝 오늘 장 시작부터 마감까지 많이 오른 순서")
        for i, (_, r) in enumerate(up.head(5).iterrows(), 1):
            out.append(f"{i}. {r['name']} {pct(r.r1d)}" + (f"  ({theme_names(r)})" if theme_names(r) else ""))
    down = sub[sub.r1d < 0].tail(1)
    if not down.empty:
        r = down.iloc[0]
        out.append(f"\n📉 가장 많이 내린 곳: {r['name']} {pct(r.r1d)}")
    n_up, n_all = int((sub.r1d > 0).sum()), len(sub)
    out.append(f"\n오른 곳 {n_up}개 / 전체 {n_all}개")
    out.append("※ 하루 움직임은 운도 커요. 참고만 하세요.")
    return "\n".join(out)


def theme_names(r):
    t = str(r.get("theme", "") or "")
    names = [THEME_NAME[k] for k in t.split("|") if k and k not in ("kr", "us") and k in globals().get("THEME_NAME", {})]
    return ", ".join(names)


def send_ntfy(title, message, click_url):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC 이 없어서 알림은 건너뜁니다.")
        return
    payload = {"topic": topic, "title": title, "message": message,
               "tags": ["chart_with_upwards_trend"]}
    if click_url:
        payload["click"] = click_url
        payload["actions"] = [{"action": "view", "label": "대시보드 열기", "url": click_url}]
    r = requests.post("https://ntfy.sh/", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, timeout=20)
    print("ntfy 전송:", r.status_code)


# ───────────────────────── 6. 웹 대시보드 ─────────────────────────

E = html.escape


def item_html(r, metric):
    hot = "<i>🔥 너무 빨리 오름</i>" if r.overheated else ""
    return (f"<li data-mk='{r.market}' data-th='{th(r)}'><div><b>{E(r['name'])}</b>"
            f"<span class='tk'>{FLAG.get(r.market, '')} {E(short_ticker(r.ticker))} · {E(r.category)}</span></div>"
            f"<span class='m'>{metric}{hot}</span></li>")


def card(title, desc, rows, metric_fn):
    items = "".join(item_html(r, metric_fn(r)) for _, r in rows.iterrows())
    return (f"<section class='card'><h2>{title}</h2><p>{desc}</p><ol class='flt'>{items}"
            f"<li class='empty'>오늘은 여기에 맞는 ETF가 없어요</li></ol></section>")


def updown(v, text=None):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    return f"<span class='{cls}'>{text or pct(v)}</span>"


def recent_panel(sub, mk):
    p = picks_all(sub)
    cards = (
        card("🏆 오늘의 추천", "요즘 잘 오르면서도 너무 출렁이지 않는 ETF예요. 100점에 가까울수록 좋아요",
             p["overall"], lambda r: f"{r.total:.0f}점") +
        card("🚀 요즘 쑥쑥 오르는 중", "최근 1~6달 동안 가격이 많이 오른 ETF예요",
             p["momentum"], lambda r: f"6달 동안 {updown(r.r6m)}") +
        card("🎢 잠깐 쉬는 중", "원래는 잘 오르던 ETF인데 요즘 잠깐 내려왔어요. 다시 오를 수도, 더 내려갈 수도 있어요",
             p["rebound"], lambda r: f"제일 비쌀 때보다 {updown(r.from_high)}") +
        card("🐢 천천히 꾸준히", "가격이 덜 출렁이고, 6달 동안 조금이라도 오른 ETF예요",
             p["stable"], lambda r: f"흔들림 {shake(r.vol60)}")
    )

    cols = [("name", "이름"), ("category", "무엇을 담았나"), ("r1d", "1일"), ("r1w", "1주"), ("r1m", "1달"), ("r3m", "3달"),
            ("r6m", "6달"), ("r12m", "1년"), ("from_high", "제일 비쌀 때보다"),
            ("vol60", "흔들림"), ("total", "점수")]

    def cell(c, r):
        v = r[c]
        if c == "name":
            return f"<td>{FLAG.get(r.market, '')} {E(str(v))}{' 🔥' if r.overheated else ''}</td>"
        if c == "category":
            return f"<td>{E(str(v))}</td>"
        if c == "vol60":
            return f"<td data-v='{v:.4f}'>{shake(v)}</td>"
        if c == "total":
            return f"<td data-v='{v:.2f}'><b>{v:.0f}</b></td>"
        if pd.isna(v):
            return "<td data-v='-999'>-</td>"
        return f"<td data-v='{v:.4f}'>{updown(v)}</td>"

    tid = f"t-{mk}"
    head = "".join(f"<th onclick=\"sortBy('{tid}',{i})\">{n}</th>" for i, (_, n) in enumerate(cols))
    body = "".join(f"<tr data-mk='{r.market}' data-th='{th(r)}'>" + "".join(cell(c, r) for c, _ in cols) + "</tr>"
                   for _, r in sub.sort_values("total", ascending=False).iterrows())
    return (f"<div class='grid'>{cards}</div>"
            f"<h3>📋 전부 보기 <span class='hint'>맨 위 제목을 누르면 순서가 바뀌어요</span></h3>"
            f"<div class='tbl'><table class='flt' id='{tid}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def price_box(r):
    """실제 가격으로 얼마 올랐는지 (예: 52,000원 → 188,700원, +136,700원)"""
    p0, p1 = r.get("p0", np.nan), r.get("p1", np.nan)
    if pd.isna(p0) or pd.isna(p1):
        return ""
    m = r["market"]
    d = p1 - p0
    sign = "+" if d > 0 else "-" if d < 0 else ""
    diff = sign + money(abs(d), m)
    return (f"<div class='won pr'><small>실제 가격</small><b>{updown(d, diff)}</b>"
            f"<small>{money(p0, m)} → {money(p1, m)}</small></div>")


def day_text(r):
    """1일: 어느 날 장 마감 기준인지"""
    m = r["market"]
    d = pd.Timestamp(r["end"])
    if r.get("daily"):
        return f"{MKT_NAME.get(m, '')} {d.month}월 {d.day}일 (전날 마감 → 이날 마감)"
    return f"{MKT_NAME.get(m, '')} {d.month}월 {d.day}일 장 시작부터 마감까지"


def why_html(text):
    return f"<span class='why'>💬 {E(text)}</span>" if text else ""


def long_panel(rows, years):
    ranked = long_rank(rows)
    _, unit, cnt, pname = PERIOD_INFO[years]
    if not rows:
        return (f"<p class='empty big'>여기엔 생긴 지 {pname}이 넘은 ETF가 없어요.<br>"
                f"생긴 지 {pname}이 안 된 ETF는 계산할 수 없어요.</p>")
    items = []
    keep = combo_keep(ranked, LONG_SHOW, lambda x: x['market'], lambda x: x.get('theme', ''))
    for i, r in enumerate(keep, 1):
        boxes = "".join(
            (f"<span class='bx n' title='{E(lb)}: 장이 쉬었어요'></span>" if b is None else
             f"<span class='bx {'u' if b > 0 else 'd' if b < 0 else 'n'}' title='{E(lb)}: {pct(b)}'></span>")
            for b, lb in zip(r["blocks"], r["labels"]))
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}")
        ups = r["ups"]
        rest = sum(b is None for b in r["blocks"])
        count_text = (day_text(r) + (f" · 하루 동안 {updown(r['total'])} 올랐어요 <span class='hint'>(1시간 단위 가격이 없어서 하루 전체로 봤어요)</span>" if r.get("daily") else f" · {len(r['blocks'])}{cnt} 중 <b>{ups}{cnt}</b> 올랐어요")) if years == "1d" else f"{len(r['blocks'])}{cnt} 중 <b>{ups}{cnt}</b> 올랐어요" + (f" ({rest}{cnt}은 장이 쉬었어요)" if rest else "") + (
            " · 1년에 평균 " + updown(r["cagr"]) if isinstance(years, int) and years > 1 else "")
        n_blocks = len(r["blocks"])
        avg = " · 1년에 평균 " + updown(r["cagr"]) if isinstance(years, int) and years > 1 else ""
        items.append(
            f"<li class='lt' data-mk='{r['market']}' data-th='{th(r)}'><div class='rk'>{medal}</div><div class='lb'>"
            f"<b>{E(r['name'])}</b>"
            f"<span class='tk'>{FLAG.get(r['market'], '')} {E(short_ticker(r['ticker']))} · {E(r['category'])}</span>"
            f"<div class='boxes'>{boxes}</div>"
            f"<span class='sm'>{count_text}</span>"
            f"{why_html(reason_for(r['ticker'], years, r.get('end')))}</div>"
            f"{price_box(r)}"
            f"<div class='won'><small>100만 원이</small><b>{won(r['total'])}</b>"
            f"<small>{updown(r['total'])}</small></div></li>")
    note = "이 기간 동안 가격이 내려간 ETF는 뺐어요. "
    return (f"<p class='how'>🟥 = 그 {unit}에 올랐어요 &nbsp; 🟦 = 그 {unit}에 내렸어요 &nbsp; ⬜ = 장이 쉬었어요 "
            f"(칸을 누르고 있으면 날짜가 보여요)<br>1일 = 그 나라 장 시작부터 마감까지 1시간씩 · 1주 = 7일 · 1달 = 4주 · 1년 = 12달. 마감 시각은 나라마다 달라요 (한국 오후 3:30 · 중국 오후 4시 · 홍콩 오후 5시 · 미국 아침 6시, 한국시간)<br>💬 = 그 기간에 왜 올랐는지 Claude가 뉴스를 보고 짧게 정리한 거예요 (틀릴 수 있어요)<br>{note}생긴 지 {pname}이 안 된 ETF는 계산에서 빠져요.</p>"
            f"<ol class='longlist'>{''.join(items)}<li class='empty'>여기에 맞는 ETF가 없어요</li></ol>"
            f"<button class='more' onclick='moreList(this)'>더보기</button>")


CSS = """
@font-face{font-family:'ChosunIlboMyungjo';src:url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_one@1.0/Chosunilbo_myungjo.woff') format('woff');font-weight:normal;font-display:swap}
:root{color-scheme:dark;--bg:#262624;--fg:#faf9f5;--sub:#a6a39a;--card:#30302e;--line:#3e3e3a;--up:#ff7b6b;--down:#6cb6ff;--acc:#d97757;--soft:#3a3a37}
*{box-sizing:border-box}body{margin:0;font-family:'ChosunIlboMyungjo',Georgia,"Noto Serif KR",serif;-webkit-font-smoothing:antialiased;background:var(--bg);color:var(--fg);-webkit-text-size-adjust:100%}
header{padding:22px 16px 6px;max-width:1100px;margin:auto}h1{font-size:23px;margin:0 0 4px}header p{color:var(--sub);margin:0;font-size:13px}
.guide{max-width:1100px;margin:10px auto 0;padding:0 16px}
.guide details{background:var(--soft);border-radius:14px;padding:12px 14px;font-size:14px;line-height:1.7}
.guide summary{font-weight:700;cursor:pointer}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);max-width:1100px;margin:auto;padding:8px 16px 6px;display:flex;flex-direction:column;gap:6px}
.row{display:flex;gap:5px}
.row.scroll{overflow-x:auto;scrollbar-width:none}.row.scroll::-webkit-scrollbar{display:none}.row.scroll button{flex:none;padding:8px 12px}.tl{flex:none;align-self:center;font-size:12px;color:var(--sub);padding:0 4px 0 2px}
.tdesc{font-size:13px;color:var(--sub);margin:2px 2px 10px}
.row button{flex:1 1 0;min-width:0;letter-spacing:-.2px;border:1px solid var(--line);background:transparent;color:var(--sub);border-radius:8px;padding:8px 4px;font-size:13px;font-family:inherit;cursor:pointer;white-space:nowrap}
.row.big button{font-size:15px;font-weight:600;padding:11px 4px;border-radius:10px}
.row button.on{background:var(--soft);border-color:var(--soft);color:var(--fg)}
main{max-width:1100px;margin:auto;padding:8px 16px 40px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px}
.card h2{font-size:17px;margin:0 0 4px}.card p{font-size:13px;color:var(--sub);margin:0 0 10px;line-height:1.5}
ol{margin:0;padding-left:20px}li{padding:8px 0;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;font-size:15px}
li:first-child{border-top:0}li div{display:flex;flex-direction:column}.tk{font-size:12px;color:var(--sub)}
.m{white-space:nowrap;font-weight:700;font-size:14px;display:flex;flex-direction:column;align-items:flex-end;gap:3px}
i{font-style:normal;background:#f59e0b22;color:#d97706;padding:1px 6px;border-radius:6px;font-size:11px;font-weight:600}
.up{color:var(--up)}.down{color:var(--down)}
.empty{color:var(--sub)}.big{font-size:15px;line-height:1.7;text-align:center;padding:30px 0}
h3{margin:26px 0 8px;font-size:16px}.hint{font-weight:400;font-size:12px;color:var(--sub)}
.tbl{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:16px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:9px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{cursor:pointer;color:var(--sub);font-weight:600;position:sticky;top:0;background:var(--card)}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
.how{font-size:13px;color:var(--sub);line-height:1.6;margin:4px 0 10px}
.longlist .lt.hide{display:none}.longlist.open .lt.hide{display:flex}
.more{display:block;width:100%;margin:8px 0 0;padding:11px;border:1px dashed var(--line);border-radius:12px;background:transparent;color:var(--sub);font:inherit;font-size:14px;cursor:pointer}
.longlist{list-style:none;padding:0;background:var(--card);border:1px solid var(--line);border-radius:16px}
.lt{padding:12px 14px;align-items:center}.rk{width:30px;flex:0 0 30px;font-size:18px;font-weight:700;text-align:center}
.lb{flex:1;min-width:0;gap:2px}.lb .boxes{display:flex;flex-direction:row;flex-wrap:wrap;gap:3px;margin:5px 0 3px}
.bx{width:14px;height:14px;border-radius:3px;display:inline-block}.bx.u{background:var(--up)}.bx.d{background:var(--down)}.bx.n{background:var(--line)}
.sm{font-size:13px;color:var(--sub)}.sm b{color:var(--fg)}
.why{display:block;font-size:13px;line-height:1.5;color:var(--fg);background:var(--soft);border-radius:8px;padding:6px 9px;margin-top:6px}
.won{text-align:right;flex:0 0 auto;display:flex;flex-direction:column}.won b{font-size:17px}.won small{font-size:11px;color:var(--sub)}
.won.pr{border-right:1px solid var(--line);padding-right:10px;margin-right:2px}.won.pr b{font-size:15px}.won.pr small:last-child{white-space:nowrap}
@media (max-width:480px){.lt{flex-wrap:wrap}.lt .lb{flex:1 1 calc(100% - 40px)}.lt .won{flex:1 1 40%;margin-top:6px;text-align:left;padding-left:38px}.lt .won.pr{border-right:0}}
.view,.mk,.yr{display:none}.view.on,.mk.on,.yr.on{display:block}
.demo{background:#f59e0b;color:#111;padding:8px 16px;text-align:center;font-weight:600}
footer{color:var(--sub);font-size:12px;line-height:1.7;margin-top:26px}
"""

JS = """
(function(){function chk(){fetch('version.json?t='+Date.now(),{cache:'no-store'}).then(r=>r.ok?r.json():null).then(v=>{
 if(v&&v.build&&v.build!==BUILD){const u=new URL(location.href);if(u.searchParams.get('v')===v.build)return;
 u.searchParams.set('v',v.build);location.replace(u.toString());}}).catch(()=>{});}
chk();document.addEventListener('visibilitychange',()=>{if(!document.hidden)chk();});window.addEventListener('pageshow',e=>{if(e.persisted)chk();});})();
function moreList(b){const l=b.previousElementSibling;if(l)l.classList.add('open');filt();}
let dir={};function sortBy(id,i){const tb=document.querySelector('#'+id+' tbody');const rows=[...tb.rows];const k=id+i;dir[k]=!dir[k];
rows.sort((a,b)=>{const x=a.cells[i],y=b.cells[i];const vx=x.dataset.v??x.textContent,vy=y.dataset.v??y.textContent;
const nx=parseFloat(vx),ny=parseFloat(vy);const c=isNaN(nx)?String(vx).localeCompare(vy,'ko'):nx-ny;return dir[k]?-c:c;});
rows.forEach(r=>tb.appendChild(r));}
const st={v:'recent',m:'all',y:'5',t:''};
function ok(e){return (st.m==='all'||e.dataset.mk===st.m.toUpperCase())&&(!st.t||(e.dataset.th||'').split('|').includes(st.t));}
function filt(){
 document.querySelectorAll('ol.flt').forEach(ol=>{let n=0;ol.querySelectorAll('li[data-mk]').forEach(li=>{const s=ok(li)&&n<CARD_N;li.style.display=s?'':'none';if(s)n++;});const em=ol.querySelector('.empty');if(em)em.style.display=n?'none':'';});
 let rows=0;document.querySelectorAll('table.flt tbody tr').forEach(tr=>{const s=ok(tr);tr.style.display=s?'':'none';if(s)rows++;});
 document.querySelectorAll('ol.longlist').forEach(ol=>{const open=ol.classList.contains('open');let n=0,shown=0;
  ol.querySelectorAll('li.lt').forEach(li=>{const m=ok(li);if(m)n++;const s=m&&n<=(open?LONG_N:FIRST_N);li.style.display=s?'flex':'none';
   if(s){shown++;const r=li.querySelector('.rk');r.textContent=n==1?'🥇':n==2?'🥈':n==3?'🥉':n;}});
  const total=Math.min(n,LONG_N);const b=ol.nextElementSibling;
  if(b&&b.classList.contains('more')){if(!open&&total>FIRST_N){b.style.display='';b.textContent='더보기 ('+(total-FIRST_N)+'개 더 · '+total+'위까지)';}else b.style.display='none';}
  const em=ol.querySelector('.empty');if(em)em.style.display=n?'none':'';});
 document.querySelectorAll('.chips').forEach(c=>{let n=0;c.querySelectorAll('.chip').forEach(x=>{const s=ok(x);x.style.display=s?'':'none';if(s)n++;});const sec=c.closest('section');if(sec)sec.style.display=n?'':'none';});
 const d=document.getElementById('tdesc');if(d)d.textContent=(TDESC[st.t]?TDESC[st.t]+' · ':'')+rows+'개';}
function set(k,val){if(k==='t'&&st.t===val)val='';st[k]=val;if(k!=='y'&&k!=='v')document.querySelectorAll('ol.longlist.open').forEach(o=>o.classList.remove('open'));draw();}
function draw(){
 document.querySelectorAll('[data-set]').forEach(b=>{const [k,v]=b.dataset.set.split(':');b.classList.toggle('on',st[k]===v);});
 document.querySelectorAll('.view').forEach(e=>e.classList.toggle('on',e.dataset.v===st.v));
 document.querySelectorAll('.yr').forEach(e=>e.classList.toggle('on',e.dataset.y===st.y));
 document.getElementById('yrow').style.display=st.v==='long'?'flex':'none';
 filt();
 document.querySelectorAll('.row.scroll').forEach(r=>{const b=r.querySelector('button.on');if(b)r.scrollLeft=b.offsetLeft-r.offsetLeft-(r.clientWidth-b.offsetWidth)/2;});
 history.replaceState(null,'','#'+st.v+'-'+st.m+'-'+st.y+(st.t?'-'+st.t:''));}
(function(){const h=location.hash.slice(1).split('-');if(h[0]==='recent'||h[0]==='long')st.v=h[0];
const isT=x=>x&&document.querySelector('[data-set="t:'+x+'"]');
if(h[1]&&document.querySelector('[data-set="m:'+h[1]+'"]'))st.m=h[1];else if(isT(h[1]))st.t=h[1];
if(h[2])st.y=h[2];if(isT(h[3]))st.t=h[3];draw();})();
"""


KR_TO_HK = re.compile(r"항셍|홍콩|HSCEI|차이나H|Hang\s?Seng", re.I)
KR_TO_CN = re.compile(r"차이나|중국|China|CSI|과창판|심천|A50", re.I)


def kr_link(name):
    """한국에 상장된 중국·홍콩 투자 ETF → 'HK' 또는 'CN' (중국·홍콩 탭에도 보여주려고)"""
    name = str(name)
    if re.search(r"ex[\s-]?China", name, re.I):     # '중국 제외' ETF
        return None
    if KR_TO_HK.search(name):
        return "HK"
    if KR_TO_CN.search(name):
        return "CN"
    return None


def tab_list(df):
    """탭: 전체 · 나라별 · (선택) 테마별"""
    tabs = [("all", "전체", None, None, "")]
    tabs += [(c.lower(), f"{f} {n}", "m", c, "") for c, f, n in MARKETS if (df.market == c).any()]
    tabs += [(k, n, "t", k, d) for k, n, d in THEMES if not in_theme(df, k).empty]
    return tabs


def tab_df(df, kind, val):
    if kind == "m":
        if val in ("CN", "HK"):
            return df[(df.market == val) | ((df.market == "KR") & (df["name"].map(kr_link) == val))]
        return df[df.market == val]
    if kind == "t":
        return in_theme(df, val)
    return df


def tab_row(r, kind, val):
    if kind == "m":
        if val in ("CN", "HK") and r["market"] == "KR":
            return kr_link(r["name"]) == val
        return r["market"] == val
    if kind == "t":
        return val in str(r.get("theme", "") or "").split("|")
    return True


def tab_desc(d, sub):
    return f"<p class='tdesc'>{E(d)} · {len(sub)}개</p>" if d else ""


def dashboard(df, long, today, demo, build=""):
    banner = "<div class='demo'>⚠ 가짜(데모) 데이터로 만든 미리보기예요</div>" if demo else ""
    tabs = tab_list(df)
    recent = f"<p class='tdesc' id='tdesc'></p>{recent_panel(df, 'all')}"
    long_html = "".join(f"<div class='yr' data-y='{y}'>{long_panel(long[y], y)}</div>" for y in PERIODS)
    tdesc = json.dumps({k: d for k, _, kind, _, d in tabs if kind == "t"}, ensure_ascii=False)

    vbtn = ("<button data-set='v:recent' onclick=\"set('v','recent')\">📅 요즘 추천</button>"
            "<button data-set='v:long' onclick=\"set('v','long')\">🌱 오래 꾸준히 오른</button>")
    mbtn = "".join(f"<button data-set='m:{k}' onclick=\"set('m','{k}')\">{lb}</button>"
                   for k, lb, kind, _, _ in tabs if kind != "t")
    tbtn = "<span class='tl'>테마</span>" + "".join(
        f"<button data-set='t:{k}' onclick=\"set('t','{k}')\">{lb}</button>"
        for k, lb, kind, _, _ in tabs if kind == "t")
    ybtn = "".join(f"<button data-set='y:{y}' onclick=\"set('y','{y}')\">{PERIOD_INFO[y][0]}</button>"
                   for y in PERIODS)
    counts = " · ".join(f"{f} {int((df.market == c).sum())}개" for c, f, _ in MARKETS
                        if (df.market == c).any())

    guide = """
<div class="guide"><details><summary>👀 이 페이지 보는 법 (눌러서 열기)</summary>
<b>주식</b>은 회사를 아주 잘게 나눈 조각이에요. 조각을 사면 그 회사의 주인이 조금 되는 거예요.<br>
<b>ETF</b>는 여러 회사의 조각을 <b>한 바구니에 담은 꾸러미</b>예요. 하나만 사도 여러 회사를 한꺼번에 가진 셈이라,
회사 하나만 살 때보다 덜 출렁여요.<br>
기본은 <b>전체 · 나라별</b>로 보고, 나라를 고른 채로 그 아래 <b>테마</b> 버튼(반도체·배당·채권처럼 비슷한 ETF 묶음)을 누르면 그 나라의 그 테마 순위만 볼 수 있어요. 테마를 한 번 더 누르면 해제돼서 그 나라 전체 순위가 보여요. 테마 버튼은 옆으로 밀면 더 있어요.<br>
점수는 100점에 가까울수록 좋아요.<br>
<b>🇨🇳 중국 · 🇭🇰 홍콩 탭</b>에는 그 나라에 상장된 ETF와 함께, 중국·홍콩에 투자하는 <b>한국 상장 ETF(🇰🇷 표시)</b>도 나와요. 🇰🇷 표시는 한국 증권 계좌로 바로 살 수 있어요.<br><br>
<b>📅 요즘 추천</b> (최근 몇 달 동안의 가격을 보고 골랐어요)<br>
&nbsp;&nbsp;🏆 오늘의 추천: 잘 오르면서 너무 출렁이지 않는 것<br>
&nbsp;&nbsp;🚀 쑥쑥 오르는 중: 요즘 가격이 많이 오른 것<br>
&nbsp;&nbsp;🎢 잠깐 쉬는 중: 잘 오르다가 요즘 잠깐 내려온 것<br>
&nbsp;&nbsp;🐢 천천히 꾸준히: 가격이 덜 출렁이는 것<br><br>
<b>표에 나오는 말</b><br>
&nbsp;&nbsp;<b>무엇을 담았나</b>: 바구니에 어떤 회사들이 들어 있는지 (예: 반도체 회사들)<br>
&nbsp;&nbsp;<b>흔들림</b>: 가격이 하루하루 얼마나 크게 오르내리는지 (작음 · 보통 · 큼)<br>
&nbsp;&nbsp;<b>제일 비쌀 때보다</b>: 지난 1년 중 가장 비쌌던 날보다 지금 얼마나 싸졌는지<br>
&nbsp;&nbsp;<span class="up">빨간 숫자</span>는 올랐다는 뜻, <span class="down">파란 숫자</span>는 내렸다는 뜻이에요<br><br>
<b>🌱 오래 꾸준히 오른</b>: 1주~20년 동안 기간마다 올랐는지 색깔 칸으로 보여줘요.<br>
<b>🔥 너무 빨리 오름</b>: 짧은 시간에 많이 올라서, 잠깐 내려갈 수도 있으니 조심
</details></div>"""

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark"><meta name="theme-color" content="#262624">
<title>ETF 추천 대시보드</title><style>{CSS}</style></head><body>{banner}
<header><h1>📊 오늘의 ETF 추천</h1><p>{today} 업데이트 · {counts} ETF를 살펴봤어요</p></header>
{guide}
<nav class="bar"><div class="row big">{vbtn}</div><div class="row">{mbtn}</div><div class="row scroll">{tbtn}</div>
<div class="row" id="yrow">{ybtn}</div></nav>
<main>
<div class="view" data-v="recent">{recent}</div>
<div class="view" data-v="long">{long_html}</div>
<footer>
💡 ETF가 나눠주는 돈(배당금)도 더해서 계산했고, 살 때 내는 수수료와 세금은 빼지 않았어요.<br>
'100만 원이 → 얼마'는 그 기간 처음에 100만 원어치 샀다면 지금 얼마가 됐을지 보여줘요.<br>
⚠ 지나간 숫자로 계산한 참고용이에요. 앞으로도 똑같이 오른다는 약속도, 사라는 뜻도 아니에요.
</footer></main>
<script>const BUILD="{build}";const TDESC={tdesc};const CARD_N={TOP_N},FIRST_N={FIRST_SHOW},LONG_N={LONG_SHOW};{JS}</script></body></html>"""


# ───────────────────────── 7. 실행 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="가짜 데이터로 테스트")
    args = ap.parse_args()

    universe = load_universe()
    tickers = universe.ticker.tolist()
    closes = demo_prices(tickers) if args.demo else fetch_prices(tickers)

    (demo_intraday(closes) if args.demo else fetch_intraday(tickers))
    df = build_table(universe, closes)
    df = apply_intraday(df)
    long = build_long(universe, closes)
    today = datetime.now(KST).strftime("%Y-%m-%d")

    build = datetime.now(KST).strftime("%Y%m%d%H%M%S")
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(dashboard(df, long, today, args.demo, build))
    with open(os.path.join(DOCS, "version.json"), "w", encoding="utf-8") as f:
        json.dump({"build": build}, f)
    df.to_csv(os.path.join(DOCS, "latest.csv"), index=False, encoding="utf-8-sig")

    close = os.environ.get("CLOSE_MARKET", "").upper()   # 장 마감 때 실행하면 KR/CN/HK
    msg = close_message(df, close, today) if close else phone_message(df, long, today)
    print(msg)

    url = os.environ.get("DASHBOARD_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not url and repo:
        owner, name = repo.split("/")
        url = f"https://{owner}.github.io/{name}/"
    if url:   # 알림에서 열 때 예전에 저장된 화면 대신 항상 새 화면이 나오게
        url = f"{url}{'&' if '?' in url else '?'}v={build}"
    if os.environ.get("GITHUB_EVENT_NAME") == "push" or os.environ.get("SKIP_ALERT"):
        print("코드·이유 파일을 고쳐서 다시 만든 거라 알림은 보내지 않아요.")
    elif not args.demo:
        if close:
            if msg:
                send_ntfy(f"{FLAG.get(close, '')} {MKT_NAME.get(close, close)}장 마감 · 오늘 많이 오른 ETF", msg,
                          f"{url}#long-{close.lower()}-1d" if url else url)
        else:
            send_ntfy("오늘의 ETF 추천", msg, url)


if __name__ == "__main__":
    sys.exit(main())
