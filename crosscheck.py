"""
야후 파이낸스 가격을 다른 사이트와 비교해서 틀린 걸 고치거나 빼요.

  🇰🇷 한국(주식·ETF): 네이버 증권 일별 시세
  🇺🇸 미국 · 🇯🇵 일본 · 🇭🇰 홍콩 · 🇨🇳 중국 · 🇻🇳 베트남: 네이버 증권 해외주식 일별 시세
  (영국·독일·인도네시아·싱가포르·캐나다는 무료로 비교할 사이트가 마땅치 않아 야후 자료 안에서만 점검)

하는 일
  1. 마지막 마감 가격이 2% 넘게 다르면 → 다른 사이트 가격으로 고쳐요
  2. 1일·1주·1달·1년 수익률이 크게 다르면 → 그 기간 순위에서 이번엔 빼요
  3. 야후가 이미 끝난 최신 장날을 빼먹으면 → 네이버 값으로 채워요 (최대 3일)
  4. 한국 ETF인데 두 기간 넘게 크게 다르면 → 야후 가격이 망가진 것으로 보고 최근 1년 남짓을 네이버 값으로 바꿔요
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
import pricecache
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
WORLD = {"US", "JP", "HK", "CN", "VN"}   # 네이버 해외주식으로 비교할 수 있는 나라
FAILS = {}   # 사이트별 연속 실패 수 (계속 실패하면 이번 실행에선 그만 물어봄)
ROTATE = 150                 # 해외는 한 번에 이만큼씩 돌아가며 비교 (+ 크게 움직인 곳은 항상)
BAD = {}                     # 티커 → 순위에서 뺄 기간 {"1d","1w","1m",1}
OPEN = {}                    # 티커 → (날짜, 거래소 공식 시작 가격) — 1일 '장 시작→마감' 계산용
REPORT = {"source_ok": 0, "source_none": 0, "fixed_last": [], "excluded": [], "no_source_markets": [],
          "added_days": [], "replaced": []}
MKT_TZ = {"US": "America/New_York", "JP": "Asia/Tokyo", "HK": "Asia/Hong_Kong", "CN": "Asia/Shanghai",
          "VN": "Asia/Ho_Chi_Minh", "KR": "Asia/Seoul"}
MAX_ADD_DAYS = 3             # 야후가 빼먹은 최신 날은 최대 이만큼만 네이버 값으로 채움


def _with_tz(idx, like):
    """새 날짜들을 야후 가격 표와 같은 시간대 모양으로"""
    idx = pd.DatetimeIndex(idx)
    tz = getattr(like.index, "tz", None)
    return idx.tz_localize(tz) if tz is not None and idx.tz is None else idx


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# ───────── 다른 사이트에서 받아오기 ─────────

def naver_series(code):
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y%m%d")
    url = (f"https://api.finance.naver.com/siseJson.naver?symbol={code}&requestType=1"
           f"&startTime={start}&endTime={end}&timeframe=day")
    txt = requests.get(url, headers=UA, timeout=15).text.strip()
    rows = ast.literal_eval(txt.replace("\n", ""))
    data = [(pd.Timestamp(str(r[0])), float(r[4])) for r in rows[1:] if r and r[4]]
    last = [r for r in rows[1:] if r and r[4]]
    if last and last[-1][1]:
        OPEN[code] = (pd.Timestamp(str(last[-1][0])), float(last[-1][1]))
    return pd.Series(dict(data)).sort_index() if data else None


def naver_world_codes(ticker, mk):
    """야후 티커 → 네이버 해외주식 코드 후보"""
    if mk == "US":
        base = ticker
        if "-" in base:   # BRK-B → BRKb
            a, c = base.split("-", 1)
            base = a + c.lower()
        return [base + ".O", base, base + ".K"]   # 나스닥 · 뉴욕 · 기타
    if mk == "VN":
        return [ticker.replace(".VN", ".HM"), ticker.replace(".VN", ".HN")]
    return [ticker]   # 일본 .T · 홍콩 .HK · 중국 .SS/.SZ 는 그대로


def naver_world_series(ticker, mk):
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=400)
    for code in naver_world_codes(ticker, mk):
        url = (f"https://api.stock.naver.com/chart/foreign/item/{code}/day?"
               f"startDateTime={start:%Y%m%d}0000&endDateTime={end:%Y%m%d}2359")
        r = requests.get(url, headers=UA, timeout=12)
        if not r.ok:
            continue
        data = r.json()
        if isinstance(data, list) and data:
            s = pd.Series({pd.Timestamp(str(x["localDate"])): float(x["closePrice"]) for x in data
                           if x.get("closePrice")}).sort_index()
            if len(s):
                lastx = max((x for x in data if x.get("closePrice")), key=lambda x: str(x["localDate"]))
                if lastx.get("openPrice"):
                    OPEN[ticker] = (pd.Timestamp(str(lastx["localDate"])), float(lastx["openPrice"]))
                return s
    return None


def source(ticker, mk):
    kind = "kr" if mk == "KR" else "world" if mk in WORLD else None
    if kind is None or FAILS.get(kind, 0) >= 15:   # 15번 연속 실패하면 그 사이트는 이번엔 쉬기
        return None
    try:
        s = naver_series(ticker.split(".")[0]) if kind == "kr" else naver_world_series(ticker, mk)
        if kind == "kr" and ticker.split(".")[0] in OPEN:
            OPEN[ticker] = OPEN.pop(ticker.split(".")[0])
    except Exception:
        s = None
    FAILS[kind] = 0 if s is not None else FAILS.get(kind, 0) + 1
    return s


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
                                         - WORLD - {"KR", ""})

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
    foreign = [t for t in tickers if mkt_of.get(t) in WORLD]
    pick = [t for t in tickers if mkt_of.get(t) in ("KR", "US")]   # 한국·미국은 매번 전부 (알림에 바로 쓰여서)
    pick += [t for t in foreign if big_move(t)]
    # 일본·홍콩·중국·베트남은 개수가 적어서 매번 전부 비교 (돌아가며 하면 빠진 최신 날을 못 채우는 곳이 생김 — 2026-11-14)
    pick += [t for t in foreign if mkt_of.get(t) != "US"]
    if foreign:
        start = rot % len(foreign)
        pick += (foreign[start:] + foreign[:start])[:ROTATE]
    pick = list(dict.fromkeys(pick))

    def one(t):
        time.sleep(0.15)
        return t, source(t, mkt_of.get(t))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:   # 동시에 4곳까지만 (네이버가 막지 않게)
        got = dict(ex.map(one, pick))
    took = time.time() - t0

    for t, src in got.items():
        if src is None or len(src) < 3:
            REPORT["source_none"] += 1
            REPORT.setdefault("stale_why", []).append(f"{t}: 네이버 자료 없음")
            continue
        REPORT["source_ok"] += 1
        # 다른 사이트(네이버 해외)는 액면분할 전 가격을 그대로 둘 때가 있어(예: IWF 2026-04-29 1→4주),
        # 비교 전에 야후 쪽과 같은 방법(pricecache.fix_jumps)으로 분할 튐을 맞춰요 (2026-11-14 검증)
        try:
            src = pricecache.fix_jumps(src, 1.6, f"(비교용 네이버) {t}")
        except Exception:
            pass
        y = raw[t]
        last = y.index[-1]
        src = src.copy()
        src.index = _with_tz(src.index, y)
        if last not in src.index:
            if mkt_of.get(t) in ("CN", "HK", "JP"):
                REPORT.setdefault("stale_why", []).append(
                    f"{t}: 네이버에 야후 마지막 날({_naive(last):%m-%d}) 값이 없음 (네이버 마지막 {_naive(src.index[-1]):%m-%d})")
            continue   # 다른 사이트가 아직 그 날 가격이 없음 → 비교 안 함
        # 1) 마지막 가격: 2% 넘게 다를 때만 고침
        #    (주의: 네이버 한국 일별 가격은 장 마감 뒤 넥스트레이드 애프터마켓(~20시)까지 합친 '통합' 값이라
        #     거래소 정규장 종가(15:30)와 몇백~몇천 원 다를 수 있어요. 그래서 작은 차이는 고치지 않아요 — 2026-11-13 확인)
        sv, yv = float(src.loc[last]), float(y.iloc[-1])
        # 한국은 정규장 종가(15:30)를 쓰기로 해서(2026-11-13 사용자 결정), 통합 가격인 네이버 값으로 바꾸지 않아요
        if mkt_of.get(t) != "KR" and sv > 0 and abs(yv / sv - 1) > 0.02:
            ratio = sv / yv
            raw[t] = pd.concat([y.iloc[:-1], pd.Series([sv], index=[last])])
            c = closes[t]
            if c.index[-1] == last:
                closes[t] = pd.concat([c.iloc[:-1], pd.Series([float(c.iloc[-1]) * ratio], index=[last])])
            REPORT["fixed_last"].append(f"{names.get(t, t)} ({yv:,.2f}→{sv:,.2f})")
            y = raw[t]
        # 1-2) 야후가 빼먹은 최신 날(이미 장이 끝난 날)을 네이버 값으로 채움 (2026-11-14 1주일 돌아보기:
        #      중국·홍콩·싱가포르 ETF가 금요일 값을 잃고 목요일에 멈춰 있었음). 한국은 ETF만 있는 봇이라
        #      애프터마켓 차이가 없어서 같이 채워요. 마지막 날 값이 2% 안으로 같을 때만.
        mk = mkt_of.get(t, "")
        if mk in MKT_TZ:
            today = pd.Timestamp.now(tz=MKT_TZ[mk]).tz_localize(None).normalize()
            sv_last = float(src.loc[last])
            newer = src[(src.index > last)]
            newer = newer[[_naive(d).normalize() < today for d in newer.index]].iloc[:MAX_ADD_DAYS]
            if len(src[src.index > last]) and not len(newer):
                pass
            elif not len(newer) and _naive(last).normalize() < today - pd.Timedelta(days=1) and mk in ("CN", "HK", "JP"):
                REPORT.setdefault("stale_why", []).append(
                    f"{t}: 야후 {_naive(last):%m-%d}, 네이버 마지막 {_naive(src.index[-1]):%m-%d}")
            elif len(newer) and not (sv_last > 0 and abs(float(y.iloc[-1]) / sv_last - 1) <= 0.02):
                REPORT.setdefault("stale_why", []).append(
                    f"{t}: 마지막 겹치는 날 값 차이 {float(y.iloc[-1]) / sv_last - 1:+.1%}")
            if len(newer) and sv_last > 0 and abs(float(y.iloc[-1]) / sv_last - 1) <= 0.02:
                yv_last = float(y.iloc[-1])
                raw[t] = pd.concat([y, newer * (yv_last / sv_last)])
                c = closes.get(t)
                if c is not None and len(c) and c.index[-1] == last:
                    closes[t] = pd.concat([c, newer * (float(c.iloc[-1]) / sv_last)])
                REPORT["added_days"].append(f"{names.get(t, t)} (+{', '.join(_naive(d).strftime('%m-%d') for d in newer.index)})")
                y = raw[t]
                last = y.index[-1]
        # 2) 기간 수익률
        s2 = src[src.index <= last]
        bad = set()
        for key, days, tol in PERIODS:
            a, b = _ret(y, last, days), _ret(s2, last, days)
            if pd.isna(a) or pd.isna(b):
                continue
            if abs(a - b) > tol and abs(a - b) > 0.25 * max(abs(a), abs(b)):
                bad.add(key)
        # 3) 한국 ETF인데 두 기간 넘게 크게 다르면 야후 가격 자체가 망가진 것 → 최근 1년 남짓을 네이버 값으로 바꿈
        #    (2026-11-14: TIGER MSCI Korea TR 야후 가격이 11/13 하루에 3.76배로 튐, 실제 37,260원).
        #    ETF는 애프터마켓 거래가 없어 네이버 값 = 정규장 종가. 그 앞 옛날 야후 가격은 이어지는 날 비율로 맞춰요.
        if mk == "KR" and len(bad) >= 2 and len(s2) >= 200:
            first = s2.index[0]
            old_part = y[y.index < first]
            if len(old_part) and first in y.index and float(y.loc[first]) > 0:
                old_part = old_part * (float(s2.loc[first]) / float(y.loc[first]))
            raw[t] = pd.concat([old_part, s2])
            closes[t] = raw[t].copy()
            REPORT["replaced"].append(f"{names.get(t, t)} (야후 가격이 이상해서 최근 {len(s2)}일을 네이버 값으로)")
            y, bad = raw[t], set()
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
          f"순위에서 뺀 곳 {len(REPORT['excluded'])}곳 · 비교 자료 없음 {REPORT['source_none']}곳 · {took:.0f}초")
    if REPORT["added_days"]:
        print(f"야후가 빼먹은 최신 날을 네이버 값으로 채운 곳 {len(REPORT['added_days'])}:", ", ".join(REPORT["added_days"][:15]))
    if REPORT.get("stale_why"):
        print("최신 날을 못 채운 곳(이유):", " / ".join(REPORT["stale_why"][:20]))
    if REPORT["replaced"]:
        print("야후 가격을 네이버 값으로 바꾼 곳:", ", ".join(REPORT["replaced"]))
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
    t = (f"🔎 야후 가격을 다른 사이트(네이버 증권: 한국·미국·일본·홍콩·중국·베트남)와 비교했어요: "
         f"{REPORT['source_ok']}곳 확인")
    if REPORT["fixed_last"]:
        t += f" · 마지막 가격을 고친 곳 {len(REPORT['fixed_last'])}곳"
    if REPORT["excluded"]:
        t += " · 숫자가 서로 달라 이번엔 순위에서 뺀 곳: " + ", ".join(esc(x) for x in REPORT["excluded"][:10])
        if len(REPORT["excluded"]) > 10:
            t += f" 외 {len(REPORT['excluded']) - 10}곳"
    if REPORT.get("stale_excluded"):
        t += (" · 최신 장날 가격을 확인할 수 없어 이번엔 뺀 곳: "
              + ", ".join(esc(x) for x in REPORT["stale_excluded"][:10])
              + (f" 외 {len(REPORT['stale_excluded']) - 10}곳" if len(REPORT["stale_excluded"]) > 10 else ""))
    if REPORT["no_source_markets"]:
        t += f" · 비교할 사이트가 없는 나라({', '.join(REPORT['no_source_markets'])})는 야후 자료 안에서만 점검했어요"
    return t + "<br>"
