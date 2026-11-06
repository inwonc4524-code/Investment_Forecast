"""
ETF 추천 봇
- etfs.csv 에 적힌 한국/미국 ETF의 가격을 받아와서
  ① 상승 흐름(모멘텀) ② 반등 후보(과매도) ③ 안정형 세 가지 기준으로 점수를 매기고
- 휴대폰(ntfy)으로 요약을 보내고, docs/index.html 대시보드를 만듭니다.

실행:
  python etf_bot.py          # 실제 데이터 (GitHub Actions에서 자동 실행)
  python etf_bot.py --demo   # 가짜 데이터로 동작 확인 (인터넷 없이)

※ 이 봇의 결과는 과거 가격으로 계산한 참고 지표일 뿐, 투자 권유가 아닙니다.
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
TOP_N = 3            # 각 카테고리에서 보여줄 개수
MIN_DAYS = 220       # 이보다 데이터가 짧으면(상장 1년 미만 등) 제외


# ───────────────────────── 1. 데이터 받아오기 ─────────────────────────

def load_universe():
    return pd.read_csv(os.path.join(HERE, "etfs.csv"), dtype=str)


def fetch_prices(tickers):
    """야후 파이낸스에서 2년치 수정종가를 받아옵니다."""
    import yfinance as yf

    raw = yf.download(tickers, period="2y", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    closes = {}
    for t in tickers:
        try:
            s = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            s = s.dropna()
            if len(s) > 0:
                closes[t] = s
        except Exception:
            pass
    missing = [t for t in tickers if t not in closes]
    if missing:
        print("가격을 못 받은 종목:", ", ".join(missing))
    return closes


def demo_prices(tickers, seed=7):
    """인터넷 없이 테스트할 때 쓰는 가짜 가격."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=datetime.now(KST).date(), periods=500)
    closes = {}
    for t in tickers:
        drift = rng.normal(0.0003, 0.0007)
        vol = rng.uniform(0.004, 0.025)
        r = rng.normal(drift, vol, len(idx))
        r[-25:] += rng.normal(0, 0.004)  # 최근 한 달은 제각각 흔들리게
        closes[t] = pd.Series(10000 * np.exp(np.cumsum(r)), index=idx)
    return closes


# ───────────────────────── 2. 지표 계산 ─────────────────────────

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
    """0~100 점으로 줄 세우기 (높을수록 좋음)."""
    r = series.rank(pct=True, ascending=ascending)
    return (r * 100).round(1)


def score(df):
    # ① 모멘텀: 1·3·6개월 수익률 가중 평균, 200일선 위에 있어야 후보
    df["mom_raw"] = 0.2 * df.r1m + 0.4 * df.r3m + 0.4 * df.r6m
    df["mom_score"] = pct_rank(df.mom_raw)
    df.loc[~df.above_ma200, "mom_score"] *= 0.5

    # ② 반등 후보: 장기 추세(200일선)는 오르는 중인데 단기로 많이 빠진 것
    df["rebound_score"] = (0.5 * pct_rank(df.rsi14, ascending=False)
                           + 0.5 * pct_rank(df.from_high, ascending=False)).round(1)
    df["rebound_ok"] = df.ma200_rising & ((df.rsi14 < 40) | (df.from_high <= -0.08))

    # ③ 안정형: 변동성·최대낙폭이 작고, 6개월 수익률은 플러스
    df["stab_score"] = (0.6 * pct_rank(df.vol60, ascending=False)
                        + 0.4 * pct_rank(df.mdd1y)).round(1)
    df["stab_ok"] = df.r6m > 0

    # 종합: 모멘텀 + 안정성, 과열(RSI 75↑)이면 감점
    df["total"] = (0.6 * df.mom_score + 0.4 * df.stab_score).round(1)
    df.loc[df.rsi14 >= 75, "total"] -= 10
    df["overheated"] = df.rsi14 >= 70
    return df


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
        "momentum": df[df.above_ma200].nlargest(TOP_N, "mom_score"),
        "rebound": df[df.rebound_ok].nlargest(TOP_N, "rebound_score"),
        "stable": df[df.stab_ok].nlargest(TOP_N, "stab_score"),
        "overall": df[df.above_ma200].nlargest(5, "total"),
    }


# ───────────────────────── 3. 휴대폰 알림 (ntfy) ─────────────────────────

def pct(x):
    return "-" if pd.isna(x) else f"{x * 100:+.1f}%"


MARKETS = [("KR", "🇰🇷 한국"), ("US", "🇺🇸 미국"), ("CN", "🇨🇳 중국"), ("HK", "🇭🇰 홍콩")]


def short_ticker(t):
    return t.split(".")[0]


def phone_message(df, today):
    """시장별로 종합 1위 + 상승·반등·안정 각 1위를 한 줄씩."""
    def first(rows):
        return None if rows.empty else rows.iloc[0]

    out = [f"📅 {today} 기준"]
    for code, label in MARKETS:
        sub = df[df.market == code]
        if sub.empty:
            continue
        p = picks(sub)
        out.append(f"\n{label}")
        for icon, key, extra in [
            ("🏆", "overall", lambda r: f"3개월 {pct(r.r3m)}"),
            ("📈", "momentum", lambda r: f"6개월 {pct(r.r6m)}"),
            ("🔄", "rebound", lambda r: f"고점대비 {pct(r.from_high)}"),
            ("🛡", "stable", lambda r: f"변동성 {r.vol60 * 100:.0f}%"),
        ]:
            r = first(p[key])
            if r is None:
                out.append(f"{icon} 해당 없음")
            else:
                flag = " 🔥과열" if r.overheated else ""
                out.append(f"{icon} {r['name']} ({short_ticker(r.ticker)}) {extra(r)}{flag}")
    out.append("\n🏆종합 📈상승 🔄반등 🛡안정 · 전체 순위는 대시보드에서")
    out.append("※ 과거 가격 기반 참고용, 투자 권유 아님")
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


# ───────────────────────── 4. 웹 대시보드 ─────────────────────────

def card(title, desc, rows, metric_fn):
    items = "".join(
        f"<li><div><b>{html.escape(r['name'])}</b>"
        f"<span class='tk'>{html.escape(r.ticker)} · {html.escape(r.category)}</span></div>"
        f"<span class='m'>{metric_fn(r)}{' <i>과열</i>' if r.overheated else ''}</span></li>"
        for _, r in rows.iterrows()
    ) or "<li class='empty'>오늘은 조건에 맞는 종목이 없어요</li>"
    return f"<section class='card'><h2>{title}</h2><p>{desc}</p><ol>{items}</ol></section>"


def panel(sub, key):
    """한 탭(시장)의 카드 4개 + 표."""
    p = picks(sub)
    cols = [("name", "이름"), ("market", "시장"), ("category", "분류"),
            ("r1m", "1개월"), ("r3m", "3개월"), ("r6m", "6개월"), ("r12m", "1년"),
            ("rsi14", "RSI"), ("vol60", "변동성"), ("mdd1y", "최대낙폭"),
            ("from_high", "고점대비"), ("total", "종합점수")]
    pct_cols = {"r1m", "r3m", "r6m", "r12m", "vol60", "mdd1y", "from_high"}

    def cell(c, v):
        if c in pct_cols:
            cls = "up" if v > 0 else "down" if v < 0 else ""
            if c == "vol60":
                cls = ""
            return f"<td class='{cls}' data-v='{v:.4f}'>{v * 100:+.1f}%</td>" if c != "vol60" \
                else f"<td data-v='{v:.4f}'>{v * 100:.0f}%</td>"
        if c in {"rsi14", "total"}:
            return f"<td data-v='{v:.2f}'>{v:.0f}</td>"
        return f"<td>{html.escape(str(v))}</td>"

    body = "".join("<tr>" + "".join(cell(c, r[c]) for c, _ in cols) + "</tr>"
                   for _, r in sub.sort_values("total", ascending=False).iterrows())
    head = "".join(f"<th onclick=\"sortBy('t-{key}',{i})\">{n}</th>"
                   for i, (_, n) in enumerate(cols))

    cards = (
        card("🏆 종합 TOP 5", "상승 흐름과 안정성을 합친 점수 (과열 종목 감점)",
             p["overall"], lambda r: f"{r.total:.0f}점") +
        card("📈 상승 흐름", "1·3·6개월 수익률이 좋고 200일 평균선 위에 있는 ETF",
             p["momentum"], lambda r: f"6개월 {pct(r.r6m)}") +
        card("🔄 반등 후보", "장기 추세는 오르는 중인데 최근 많이 빠진 ETF",
             p["rebound"], lambda r: f"고점대비 {pct(r.from_high)} · RSI {r.rsi14:.0f}") +
        card("🛡 안정형", "출렁임(변동성)과 최대 낙폭이 작고 6개월 수익이 플러스인 ETF",
             p["stable"], lambda r: f"변동성 {r.vol60 * 100:.0f}%")
    )
    return (f"<div class='panel' id='p-{key}'><div class='grid'>{cards}</div>"
            f"<h3>ETF 표 <span class='hint'>(제목을 누르면 정렬)</span></h3>"
            f"<div class='tbl'><table id='t-{key}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div></div>")


def dashboard(df, today, demo):
    banner = "<div class='demo'>⚠ 가짜(데모) 데이터로 만든 미리보기입니다</div>" if demo else ""
    tabs_def = [("all", "전체", df)] + [
        (code.lower(), label, df[df.market == code]) for code, label in MARKETS]
    tabs_def = [t for t in tabs_def if not t[2].empty]
    tabs = "".join(f"<button data-k='{k}' onclick=\"showTab('{k}')\">{label}"
                   f"<small>{len(sub)}</small></button>" for k, label, sub in tabs_def)
    panels = "".join(panel(sub, k) for k, _, sub in tabs_def)
    counts = " · ".join(f"{label.split()[-1]} {int((df.market == c).sum())}개"
                        for c, label in MARKETS if (df.market == c).any())

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETF 추천 대시보드</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1c1f24;--sub:#6b7280;--card:#fff;--line:#e5e7eb;--up:#d63c3c;--down:#2f6fd6;--acc:#0f766e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111316;--fg:#e8eaed;--sub:#9aa0a6;--card:#1b1e22;--line:#2c3036;--up:#ff6b6b;--down:#6aa5ff;--acc:#2dd4bf}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;background:var(--bg);color:var(--fg)}}
header{{padding:24px 20px 8px;max-width:1100px;margin:auto}}h1{{font-size:22px;margin:0 0 4px}}header p{{color:var(--sub);margin:0;font-size:14px}}
main{{max-width:1100px;margin:auto;padding:12px 20px 40px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}}
.card h2{{font-size:16px;margin:0 0 4px}}.card p{{font-size:12px;color:var(--sub);margin:0 0 10px}}
ol{{margin:0;padding-left:20px}}li{{padding:7px 0;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;font-size:14px}}
li:first-child{{border-top:0}}li div{{display:flex;flex-direction:column}}.tk{{font-size:11px;color:var(--sub)}}
.m{{white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--acc);font-weight:600;font-size:13px}}
i{{font-style:normal;background:#f59e0b22;color:#d97706;padding:1px 6px;border-radius:6px;font-size:11px}}
.empty{{color:var(--sub)}}h3{{margin:28px 0 8px;font-size:16px}}
.tbl{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th{{cursor:pointer;color:var(--sub);font-weight:600;position:sticky;top:0;background:var(--card)}}
th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}.up{{color:var(--up)}}.down{{color:var(--down)}}
.demo{{background:#f59e0b;color:#111;padding:8px 20px;text-align:center;font-weight:600}}
footer{{color:var(--sub);font-size:12px;line-height:1.6;margin-top:24px}}
.hint{{font-weight:400;font-size:12px;color:var(--sub)}}
.tabs{{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;gap:5px;padding:10px 20px;max-width:1100px;margin:auto}}
.tabs button{{flex:1 1 0;min-width:0;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:8px 4px;font-size:13px;font-family:inherit;cursor:pointer;white-space:nowrap}}
@media (max-width:480px){{.tabs{{padding:10px 12px}}.tabs button small{{display:none}}}}
.tabs button small{{margin-left:5px;color:var(--sub);font-size:11px}}
.tabs button.on{{background:var(--acc);border-color:var(--acc);color:#fff}}.tabs button.on small{{color:#ffffffcc}}
.panel{{display:none}}.panel.on{{display:block}}
</style></head><body>{banner}
<header><h1>📊 오늘의 ETF 추천</h1><p>{today} 업데이트 · {counts} ETF 분석</p></header>
<nav class="tabs">{tabs}</nav>
<main>{panels}
<footer>
<b>용어</b> · RSI: 최근 14일 동안 얼마나 올랐/내렸는지(30 아래면 많이 빠진 상태, 70 위면 과열) ·
변동성: 1년 기준으로 환산한 출렁임 크기 · 최대낙폭: 최근 1년 중 고점에서 가장 많이 빠졌던 폭<br>
⚠ 이 페이지는 과거 가격으로 계산한 참고 지표이며 투자 권유가 아닙니다. 실제 투자 판단과 책임은 본인에게 있습니다.
</footer></main>
<script>
let dir={{}};function sortBy(id,i){{const tb=document.querySelector('#'+id+' tbody');const rows=[...tb.rows];const k=id+i;dir[k]=!dir[k];
rows.sort((a,b)=>{{const x=a.cells[i],y=b.cells[i];const vx=x.dataset.v??x.textContent,vy=y.dataset.v??y.textContent;
const nx=parseFloat(vx),ny=parseFloat(vy);const c=isNaN(nx)?String(vx).localeCompare(vy,'ko'):nx-ny;return dir[k]?-c:c;}});
rows.forEach(r=>tb.appendChild(r));}}
function showTab(k){{if(!document.getElementById('p-'+k))k='all';
document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id==='p-'+k));
document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.k===k));
history.replaceState(null,'','#'+k);}}
showTab(location.hash.slice(1)||'all');
</script></body></html>"""


# ───────────────────────── 5. 실행 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="가짜 데이터로 테스트")
    args = ap.parse_args()

    universe = load_universe()
    tickers = universe.ticker.tolist()
    closes = demo_prices(tickers) if args.demo else fetch_prices(tickers)

    df = build_table(universe, closes)
    today = datetime.now(KST).strftime("%Y-%m-%d")

    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(dashboard(df, today, args.demo))
    df.to_csv(os.path.join(DOCS, "latest.csv"), index=False, encoding="utf-8-sig")

    msg = phone_message(df, today)
    print(msg)

    url = os.environ.get("DASHBOARD_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")  # 예: myname/etf-bot
    if not url and repo:
        owner, name = repo.split("/")
        url = f"https://{owner}.github.io/{name}/"
    if not args.demo:
        send_ntfy("오늘의 ETF 추천", msg, url)


if __name__ == "__main__":
    sys.exit(main())
