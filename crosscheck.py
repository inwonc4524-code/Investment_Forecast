"""
야후 파이낸스 가격을 다른 사이트와 비교해서 틀린 걸 고치거나 빼요.

  🇰🇷 한국(주식·ETF): 네이버 증권 일별 시세
  🇺🇸 미국 · 🇯🇵 일본 · 🇬🇧 영국 · 🇩🇪 독일 · 🇭🇰 홍콩: Stooq 일별 시세
  (중국·베트남·인도네시아·싱가포르·캐나다는 무료로 비교할 사이트가 마땅치 않아 야후 자료 안에서만 점검)

하는 일
  1. 마지막 마감 가격이 2% 넘게 다르면 → 다른 사이트 가격으로 고쳐요
  2. 1일·1주·1달·1년 수익률이 크게 다르면 → 그 기간 순위에서 이번엔 빼요
결과는 docs/crosscheck.json 에 남기고, 대시보드 맨 아래에 한 줄로 보여줘요.
"""

import ast
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
STOOQ_SUFFIX = {"US": "us", "JP": "jp", "UK": "uk", "DE": "de", "HK": "hk"}
ROTATE = 150                 # 해외는 한 번에 이만큼씩 돌아가며 비교 (+ 크게 움직인 곳은 항상)
BAD = {}                     # 티커 → 순위에서 뺄 기간 {"1d","1w","1m",1}
REPORT = {"source_ok": 0, "source_none": 0, "fixed_last": [], "excluded": [], "no_source_markets": []}


# ───────── 다른 사이트에서 받아오기 ─────────

def naver_series(code):
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y%m%d")
    url = (f"https://api.finance.naver.com/siseJson.naver?symbol={code}&requestType=1"
           f"&startTime={start}&endTime={end}&timeframe=day")
    txt = requests.get(url, headers=UA, timeout=15).text.strip()
    rows = ast.literal_eval(txt.replace("\n", ""))
    data = [(pd.Timestamp(str(r[0])), float(r[4])) for r in rows[1:] if r and r[4]]
    return pd.Series(dict(data)).sort_index() if data else None


def stooq_series(ticker, mk):
    base = ticker.split(".")[0].lower()
    cands = [base]
    if mk == "HK":
        cands = [base.lstrip("0") or "0", base]
    for b in cands:
        url = f"https://stooq.com/q/d/l/?s={b}.{STOOQ_SUFFIX[mk]}&i=d"
        r = requests.get(url, headers=UA, timeout=15)
        if not r.ok or not r.text.startswith("Date"):
            continue
        d = pd.read_csv(io.StringIO(r.text))
        if d.empty:
            continue
        s = pd.Series(d.Close.values, index=pd.to_datetime(d.Date)).dropna()
        s = s[s.index >= pd.Timestamp.today() - pd.Timedelta(days=400)]
        if mk == "UK" and len(s):   # Stooq는 파운드, 야후는 펜스일 수 있어서 맞춤
            s = s * 100
        return s if len(s) else None
    return None


def source(ticker, mk):
    try:
        if mk == "KR":
            return naver_series(ticker.split(".")[0])
        if mk in STOOQ_SUFFIX:
            return stooq_series(ticker, mk)
    except Exception:
        return None
    return None


# ───────── 비교 ─────────

def _ret(s, end, days):
    if days == "1d":
        if len(s) < 2:
            return np.nan
        return s.iloc[-1] / s.iloc[-2] - 1
    start = end - pd.Timedelta(days=days)
    if s.index[0] > start:
        return np.nan
    return s.iloc[-1] / s.asof(start) - 1


PERIODS = [("1d", "1d", 0.05), ("1w", 7, 0.08), ("1m", 28, 0.12), (1, 365, 0.25)]   # (기간, 날짜, 허용 차이)


def run(closes, raw, mkt_of, names=None, state_path=None):
    """closes: 수익률용(배당 포함) / raw: 실제 가격. 틀린 마지막 가격은 고치고, 수익률이 크게 다른 기간은 BAD에 기록."""
    names = names or {}
    tickers = [t for t in closes if t in raw and len(raw[t]) > 2]
    REPORT["no_source_markets"] = sorted({mkt_of.get(t, "") for t in tickers}
                                         - set(STOOQ_SUFFIX) - {"KR", ""})

    # 비교할 곳 고르기: 한국은 전부, 해외는 크게 움직인 곳 + 돌아가며 ROTATE개
    def big_move(t):
        s = raw[t]
        r1 = _ret(s, s.index[-1], "1d")
        r7 = _ret(s, s.index[-1], 7)
        r28 = _ret(s, s.index[-1], 28)
        r365 = _ret(s, s.index[-1], 365)
        return any(abs(x) > th for x, th in ((r1, .08), (r7, .2), (r28, .35), (r365, 1.5)) if not pd.isna(x))

    rot = 0
    if state_path:
        try:
            rot = json.load(open(state_path)).get("rot", 0)
        except Exception:
            rot = 0
    foreign = [t for t in tickers if mkt_of.get(t) in STOOQ_SUFFIX]
    pick = [t for t in tickers if mkt_of.get(t) == "KR"]
    pick += [t for t in foreign if big_move(t)]
    if foreign:
        start = rot % len(foreign)
        pick += (foreign[start:] + foreign[:start])[:ROTATE]
    pick = list(dict.fromkeys(pick))

    def one(t):
        time.sleep(0.15)
        return t, source(t, mkt_of.get(t))

    with ThreadPoolExecutor(max_workers=4) as ex:
        got = dict(ex.map(one, pick))

    for t, src in got.items():
        if src is None or len(src) < 3:
            REPORT["source_none"] += 1
            continue
        REPORT["source_ok"] += 1
        y = raw[t]
        last = y.index[-1]
        if last not in src.index:
            continue   # 다른 사이트가 아직 그 날 가격이 없음 → 비교 안 함
        # 1) 마지막 가격
        sv, yv = float(src.loc[last]), float(y.iloc[-1])
        if sv > 0 and abs(yv / sv - 1) > 0.02:
            ratio = sv / yv
            raw[t] = pd.concat([y.iloc[:-1], pd.Series([sv], index=[last])])
            c = closes[t]
            if c.index[-1] == last:
                closes[t] = pd.concat([c.iloc[:-1], pd.Series([float(c.iloc[-1]) * ratio], index=[last])])
            REPORT["fixed_last"].append(f"{names.get(t, t)} ({yv:,.2f}→{sv:,.2f})")
            y = raw[t]
        # 2) 기간 수익률
        s2 = src[src.index <= last]
        bad = set()
        for key, days, tol in PERIODS:
            a, b = _ret(y, last, days), _ret(s2, last, days)
            if pd.isna(a) or pd.isna(b):
                continue
            if abs(a - b) > tol and abs(a - b) > 0.25 * max(abs(a), abs(b)):
                bad.add(key)
        if bad:
            BAD[t] = bad
            per = ",".join({"1d": "1일", "1w": "1주", "1m": "1달", 1: "1년"}[k] for k in
                           [k for k, _, _ in PERIODS if k in bad])
            REPORT["excluded"].append(f"{names.get(t, t)} ({per})")

    if state_path:
        try:
            json.dump({"rot": rot + ROTATE}, open(state_path, "w"))
        except Exception:
            pass
    print(f"다른 사이트와 비교: {REPORT['source_ok']}곳 비교 · 마지막 가격 고침 {len(REPORT['fixed_last'])}곳 · "
          f"순위에서 뺀 곳 {len(REPORT['excluded'])}곳 · 비교 자료 없음 {REPORT['source_none']}곳")
    return BAD


def save(path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(REPORT, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print("비교 기록 저장 실패:", e)


def note_html(esc=lambda x: x):
    if not REPORT["source_ok"]:
        return ""
    t = (f"🔎 야후 가격을 다른 사이트(한국: 네이버 증권, 미국·일본·영국·독일·홍콩: Stooq)와 비교했어요: "
         f"{REPORT['source_ok']}곳 확인")
    if REPORT["fixed_last"]:
        t += f" · 마지막 가격을 고친 곳 {len(REPORT['fixed_last'])}곳"
    if REPORT["excluded"]:
        t += " · 숫자가 서로 달라 이번엔 순위에서 뺀 곳: " + ", ".join(esc(x) for x in REPORT["excluded"][:10])
        if len(REPORT["excluded"]) > 10:
            t += f" 외 {len(REPORT['excluded']) - 10}곳"
    if REPORT["no_source_markets"]:
        t += f" · 비교할 사이트가 없는 나라({', '.join(REPORT['no_source_markets'])})는 야후 자료 안에서만 점검했어요"
    return t + "<br>"
