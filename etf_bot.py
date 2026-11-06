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
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

KST = timezone(timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
TOP_N = 3                    # '요즘 추천' 카드마다 보여줄 개수
LONG_SHOW = 10               # '오래 꾸준히' 목록에 보여줄 개수
MIN_DAYS = 220               # 이보다 데이터가 짧으면(상장 1년 미만 등) 제외
PERIODS = ["1w", "1m", 1, 3, 5, 10, 20]  # '오래 꾸준히' 기간
PERIOD_INFO = {   # 기간: (버튼 이름, 칸 하나의 단위, 세는 말, 기간 이름)
    "1w": ("1주", "날", "일", "1주"), "1m": ("1달", "날", "일", "1달"),
    1: ("1년", "달", "달", "1년"), 3: ("3년", "해", "번", "3년"), 5: ("5년", "해", "번", "5년"),
    10: ("10년", "해", "번", "10년"), 20: ("20년", "해", "번", "20년"),
}

MARKETS = [("KR", "🇰🇷", "한국"), ("US", "🇺🇸", "미국"),
           ("CN", "🇨🇳", "중국"), ("HK", "🇭🇰", "홍콩")]
FLAG = {c: f for c, f, _ in MARKETS}


# ───────────────────────── 1. 데이터 받아오기 ─────────────────────────

def load_universe():
    return pd.read_csv(os.path.join(HERE, "etfs.csv"), dtype=str)


def fetch_prices(tickers):
    """야후 파이낸스에서 상장 이후 전체 기간의 가격(배당 포함 수정종가)을 받아옵니다."""
    import yfinance as yf

    raw = yf.download(tickers, period="max", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    closes = {}
    for t in tickers:
        try:
            s = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            s = s.dropna()
            s = s[s > 0]
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
        "r1w": ret(s, 5),
        "r1m": ret(s, 21),
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
    if years in ("1w", "1m"):
        n = 5 if years == "1w" else 21
        if len(s) < n + 1:
            return None
        w = s.iloc[-(n + 1):]
        blocks = list((w / w.shift(1) - 1).iloc[1:])
        total = float(w.iloc[-1] / w.iloc[0] - 1)
        return {"total": total, "cagr": np.nan, "blocks": blocks,
                "labels": [d.strftime("%m월 %d일") for d in w.index[1:]],
                "ups": sum(b > 0 for b in blocks)}
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
            lt = long_term(s.dropna(), y)
            if lt:
                rows.append({**u.to_dict(), **lt})
        out[y] = rows
    return out


def long_rank(rows):
    """올라간 횟수가 많은 순 → 같으면 1년 평균 수익률 높은 순. 전체로 손해 본 건 제외."""
    good = [r for r in rows if r["total"] > 0]
    return sorted(good, key=lambda r: (r["ups"] / len(r["blocks"]), r["cagr"]), reverse=True)


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


def picks(df):
    return {
        "overall": df[df.above_ma200].nlargest(5, "total"),
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
    out.append("\n🔥 = 너무 빨리 올라서 조심")
    out.append("※ 지나간 숫자로 계산한 참고용이에요. 꼭 오른다는 뜻도, 사라는 뜻도 아니에요.")
    return "\n".join(out)


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
    return (f"<li><div><b>{E(r['name'])}</b>"
            f"<span class='tk'>{FLAG.get(r.market, '')} {E(short_ticker(r.ticker))} · {E(r.category)}</span></div>"
            f"<span class='m'>{metric}{hot}</span></li>")


def card(title, desc, rows, metric_fn):
    items = "".join(item_html(r, metric_fn(r)) for _, r in rows.iterrows()) \
        or "<li class='empty'>오늘은 여기에 맞는 ETF가 없어요</li>"
    return f"<section class='card'><h2>{title}</h2><p>{desc}</p><ol>{items}</ol></section>"


def updown(v, text=None):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    return f"<span class='{cls}'>{text or pct(v)}</span>"


def recent_panel(sub, mk):
    p = picks(sub)
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

    cols = [("name", "이름"), ("category", "무엇을 담았나"), ("r1w", "1주"), ("r1m", "1달"), ("r3m", "3달"),
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
    body = "".join("<tr>" + "".join(cell(c, r) for c, _ in cols) + "</tr>"
                   for _, r in sub.sort_values("total", ascending=False).iterrows())
    return (f"<div class='grid'>{cards}</div>"
            f"<h3>📋 전부 보기 <span class='hint'>맨 위 제목을 누르면 순서가 바뀌어요</span></h3>"
            f"<div class='tbl'><table id='{tid}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def long_panel(rows, years):
    ranked = long_rank(rows)
    _, unit, cnt, pname = PERIOD_INFO[years]
    if not rows:
        return (f"<p class='empty big'>여기엔 생긴 지 {pname}이 넘은 ETF가 없어요.<br>"
                f"생긴 지 {pname}이 안 된 ETF는 계산할 수 없어요.</p>")
    items = []
    for i, r in enumerate(ranked[:LONG_SHOW], 1):
        boxes = "".join(
            f"<span class='bx {'u' if b > 0 else 'd'}' title='{E(lb)}: {pct(b)}'></span>"
            for b, lb in zip(r["blocks"], r["labels"]))
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}")
        n_blocks = len(r["blocks"])
        avg = " · 1년에 평균 " + updown(r["cagr"]) if isinstance(years, int) and years > 1 else ""
        items.append(
            f"<li class='lt'><div class='rk'>{medal}</div><div class='lb'>"
            f"<b>{E(r['name'])}</b>"
            f"<span class='tk'>{FLAG.get(r['market'], '')} {E(short_ticker(r['ticker']))} · {E(r['category'])}</span>"
            f"<div class='boxes'>{boxes}</div>"
            f"<span class='sm'>{n_blocks}{cnt} 중 <b>{r['ups']}{cnt}</b> 올랐어요{avg}</span></div>"
            f"<div class='won'><small>100만 원이</small><b>{won(r['total'])}</b>"
            f"<small>{updown(r['total'])}</small></div></li>")
    lost = len(rows) - len(ranked)
    note = f"이 기간 동안 가격이 내려간 ETF {lost}개는 뺐어요. " if lost else ""
    return (f"<p class='how'>🟥 = 그 {unit}에 올랐어요 &nbsp; 🟦 = 그 {unit}에 내렸어요 "
            f"(칸을 누르고 있으면 날짜가 보여요)<br>{note}생긴 지 {pname}이 안 된 ETF는 계산에서 빠져요.</p>"
            f"<ol class='longlist'>{''.join(items) or '<li class=empty>여기에 맞는 ETF가 없어요</li>'}</ol>")


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
.longlist{list-style:none;padding:0;background:var(--card);border:1px solid var(--line);border-radius:16px}
.lt{padding:12px 14px;align-items:center}.rk{width:30px;flex:0 0 30px;font-size:18px;font-weight:700;text-align:center}
.lb{flex:1;min-width:0;gap:2px}.lb .boxes{display:flex;flex-direction:row;flex-wrap:wrap;gap:3px;margin:5px 0 3px}
.bx{width:14px;height:14px;border-radius:3px;display:inline-block}.bx.u{background:var(--up)}.bx.d{background:var(--down)}
.sm{font-size:13px;color:var(--sub)}.sm b{color:var(--fg)}
.won{text-align:right;flex:0 0 auto}.won b{font-size:17px}.won small{font-size:11px;color:var(--sub)}
.view,.mk,.yr{display:none}.view.on,.mk.on,.yr.on{display:block}
.demo{background:#f59e0b;color:#111;padding:8px 16px;text-align:center;font-weight:600}
footer{color:var(--sub);font-size:12px;line-height:1.7;margin-top:26px}
"""

JS = """
(function(){function chk(){fetch('version.json?t='+Date.now(),{cache:'no-store'}).then(r=>r.ok?r.json():null).then(v=>{
 if(v&&v.build&&v.build!==BUILD){const u=new URL(location.href);if(u.searchParams.get('v')===v.build)return;
 u.searchParams.set('v',v.build);location.replace(u.toString());}}).catch(()=>{});}
chk();document.addEventListener('visibilitychange',()=>{if(!document.hidden)chk();});window.addEventListener('pageshow',e=>{if(e.persisted)chk();});})();
let dir={};function sortBy(id,i){const tb=document.querySelector('#'+id+' tbody');const rows=[...tb.rows];const k=id+i;dir[k]=!dir[k];
rows.sort((a,b)=>{const x=a.cells[i],y=b.cells[i];const vx=x.dataset.v??x.textContent,vy=y.dataset.v??y.textContent;
const nx=parseFloat(vx),ny=parseFloat(vy);const c=isNaN(nx)?String(vx).localeCompare(vy,'ko'):nx-ny;return dir[k]?-c:c;});
rows.forEach(r=>tb.appendChild(r));}
const st={v:'recent',m:'all',y:'5'};
function set(k,val){st[k]=val;draw();}
function draw(){
 document.querySelectorAll('[data-set]').forEach(b=>{const [k,v]=b.dataset.set.split(':');b.classList.toggle('on',st[k]===v);});
 document.querySelectorAll('.view').forEach(e=>e.classList.toggle('on',e.dataset.v===st.v));
 document.querySelectorAll('.mk').forEach(e=>e.classList.toggle('on',e.dataset.m===st.m));
 document.querySelectorAll('.yr').forEach(e=>e.classList.toggle('on',e.dataset.y===st.y));
 document.getElementById('yrow').style.display=st.v==='long'?'flex':'none';
 history.replaceState(null,'','#'+st.v+'-'+st.m+'-'+st.y);}
(function(){const h=location.hash.slice(1).split('-');if(h[0]==='recent'||h[0]==='long')st.v=h[0];
if(h[1]&&document.querySelector('[data-set="m:'+h[1]+'"]'))st.m=h[1];if(h[2])st.y=h[2];draw();})();
"""


def dashboard(df, long, today, demo, build=""):
    banner = "<div class='demo'>⚠ 가짜(데모) 데이터로 만든 미리보기예요</div>" if demo else ""
    markets = [("all", "전체", None)] + [(c.lower(), f"{f} {n}", c) for c, f, n in MARKETS
                                          if (df.market == c).any()]

    recent = "".join(
        f"<div class='mk' data-m='{k}'>{recent_panel(df if c is None else df[df.market == c], k)}</div>"
        for k, _, c in markets)
    long_html = ""
    for k, _, c in markets:
        inner = "".join(
            f"<div class='yr' data-y='{y}'>"
            f"{long_panel([r for r in long[y] if c is None or r['market'] == c], y)}</div>"
            for y in PERIODS)
        long_html += f"<div class='mk' data-m='{k}'>{inner}</div>"

    vbtn = ("<button data-set='v:recent' onclick=\"set('v','recent')\">📅 요즘 추천</button>"
            "<button data-set='v:long' onclick=\"set('v','long')\">🌱 오래 꾸준히 오른</button>")
    mbtn = "".join(f"<button data-set='m:{k}' onclick=\"set('m','{k}')\">{lb}</button>"
                   for k, lb, _ in markets)
    ybtn = "".join(f"<button data-set='y:{y}' onclick=\"set('y','{y}')\">{PERIOD_INFO[y][0]}</button>"
                   for y in PERIODS)
    counts = " · ".join(f"{f} {int((df.market == c).sum())}개" for c, f, _ in MARKETS
                        if (df.market == c).any())

    guide = """
<div class="guide"><details><summary>👀 이 페이지 보는 법 (눌러서 열기)</summary>
<b>주식</b>은 회사를 아주 잘게 나눈 조각이에요. 조각을 사면 그 회사의 주인이 조금 되는 거예요.<br>
<b>ETF</b>는 여러 회사의 조각을 <b>한 바구니에 담은 꾸러미</b>예요. 하나만 사도 여러 회사를 한꺼번에 가진 셈이라,
회사 하나만 살 때보다 덜 출렁여요.<br>
점수는 100점에 가까울수록 좋아요.<br><br>
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
<nav class="bar"><div class="row big">{vbtn}</div><div class="row">{mbtn}</div>
<div class="row" id="yrow">{ybtn}</div></nav>
<main>
<div class="view" data-v="recent">{recent}</div>
<div class="view" data-v="long">{long_html}</div>
<footer>
💡 ETF가 나눠주는 돈(배당금)도 더해서 계산했고, 살 때 내는 수수료와 세금은 빼지 않았어요.<br>
'100만 원이 → 얼마'는 그 기간 처음에 100만 원어치 샀다면 지금 얼마가 됐을지 보여줘요.<br>
⚠ 지나간 숫자로 계산한 참고용이에요. 앞으로도 똑같이 오른다는 약속도, 사라는 뜻도 아니에요.
</footer></main>
<script>const BUILD="{build}";{JS}</script></body></html>"""


# ───────────────────────── 7. 실행 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="가짜 데이터로 테스트")
    args = ap.parse_args()

    universe = load_universe()
    tickers = universe.ticker.tolist()
    closes = demo_prices(tickers) if args.demo else fetch_prices(tickers)

    df = build_table(universe, closes)
    long = build_long(universe, closes)
    today = datetime.now(KST).strftime("%Y-%m-%d")

    build = datetime.now(KST).strftime("%Y%m%d%H%M%S")
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(dashboard(df, long, today, args.demo, build))
    with open(os.path.join(DOCS, "version.json"), "w", encoding="utf-8") as f:
        json.dump({"build": build}, f)
    df.to_csv(os.path.join(DOCS, "latest.csv"), index=False, encoding="utf-8-sig")

    msg = phone_message(df, long, today)
    print(msg)

    url = os.environ.get("DASHBOARD_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not url and repo:
        owner, name = repo.split("/")
        url = f"https://{owner}.github.io/{name}/"
    if url:   # 알림에서 열 때 예전에 저장된 화면 대신 항상 새 화면이 나오게
        url = f"{url}{'&' if '?' in url else '?'}v={build}"
    if not args.demo:
        send_ntfy("오늘의 ETF 추천", msg, url)


if __name__ == "__main__":
    sys.exit(main())
