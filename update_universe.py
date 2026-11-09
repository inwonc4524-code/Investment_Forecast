"""
ETF 목록(etfs.csv) 자동 업데이트 — 한 달에 한 번 GitHub Actions가 실행해요.

기준
  🇰🇷 한국: 한국에 상장된 ETF 중 시가총액 상위 100개        ← 네이버 증권
  🇺🇸 미국: 미국 주요 ETF 후보 중 운용자산(모인 돈) 상위 100개 ← 야후 파이낸스
  공통: 레버리지·인버스(하루 움직임을 2배·거꾸로 따라가는 상품)와 파킹형(CD금리·머니마켓)은 뺌
  🇨🇳 중국·🇭🇰 홍콩: 자동 순위 자료가 없어서 지금 목록 그대로
지금 목록에 있는 ETF는 그대로 두고, 순위 안에 새로 들어온 ETF만 더해요. 밀려난 ETF도 지우지 않아요.

실행: python update_universe.py
"""

import csv
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "etfs.csv")
KR_TOP = 100
US_TOP = 100
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}

SKIP = re.compile(r"레버리지|인버스|2X|3X|곱버스|CD\S*금리|KOFR|SOFR|머니마켓|MMF|단기자금|전단채|초단기|자동연장|"
                  r"단기통안채|단기채권|금리액티브|Ultra|Leveraged|Inverse|Bear|Short|T-?Bill|0-3 Month|1-3 Month",
                  re.I)

# 미국 ETF 후보 (레버리지·인버스 제외) — 이 중에서 운용자산 상위 100개를 골라요
US_CANDIDATES = """
VOO SPY IVV VTI QQQ VEA VUG IEFA VTV BND AGG IEMG IWF VGT VXUS VO IJH GLD IWM VIG IJR BNDX VWO
SCHD XLK RSP ITOT VYM IWD SCHX EFA VB IVW SGOV VCIT XLF SCHF VT BIL QUAL MUB IXUS VEU SPLG VCSH
VNQ IWR TLT IAU SCHB VV SCHG MGK XLV IVE SHY USMV DGRO SMH JEPI IUSB VBR IEF VTEB VGIT BSV SPYG
XLE COWZ MBB GOVT SCHA IGSB XLY VHT ACWI DIA VBK VGSH SPDW IUSG SPYV EEM JPST USHY LQD HYG XLI
XLC XLP XLU XLB XLRE SOXX IBB XBI ARKK VNQI SLV GDX USO DBC TIP SCHP FNDX AVUV QQQM JEPQ SPLV
FTEC VOT VOE EFV IWB IWN IWO MTUM IJK IJJ VDE VPU VFH VDC VCR VIS VAW VOX PFF EMB VMBS STIP
SCHZ IGIB BIV BLV EDV TLH ICLN TAN LIT URA BOTZ ITA XAR CIBR IGV KWEB MCHI FXI EWJ EWY EWT INDA
EWZ VGK IEUR DFAC DFUS AVDV SCHM SCHV SCHE OEF MDY IWV VXF ESGU DYNF SPTM SPEM SPHQ SCHH BBJP
""".split()

THEME_WORDS = [
    (r"S&P ?500|S&P500|SPDR S&P 500|Vanguard 500", "sp500"),
    (r"반도체|Semiconductor|SOX", "semi"),
    (r"나스닥|Nasdaq|테크|Tech|빅테크|소프트웨어|Software|Internet|인터넷", "tech"),
    (r"2차전지|배터리|Battery|Lithium|리튬|전기차", "bat"),
    (r"태양광|Solar|Clean|친환경|수소", "ess"),
    (r"원자력|전력|Uranium|Nuclear|Utilities|Utility|Grid|인프라", "power"),
    (r"로봇|Robot|Robotics|\bAI\b|인공지능", "robot"),
    (r"바이오|헬스케어|Bio|Health|제약|Pharma", "bio"),
    (r"방산|조선|Defense|Aerospace", "defense"),
    (r"배당|Dividend|고배당|Income|커버드콜|Covered", "div"),
    (r"골드|금현물|금선물|KRX금|Gold|Silver|실버|원유|WTI|\bOil\b|Commodit|원자재", "gold"),
    (r"채권|국채|국고채|회사채|Treasury|Bond|Aggregate|TIPS|Muni|Corporate", "bond"),
    (r"일본|인도|중국|베트남|대만|유럽|Japan|India|China|Taiwan|Europe|Emerging|Developed|International|ex-US|EAFE", "world"),
]


def num(s):
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)) or "nan")
    except ValueError:
        return float("nan")


def themes_for(name):
    found = []
    for pat, t in THEME_WORDS:
        if re.search(pat, name, re.I) and t not in found:
            found.append(t)
    return "|".join(found[:2])


def load():
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("theme", "")
    return rows


def save(rows):
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "name", "market", "category", "theme"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})


# ───────── 한국 ETF: 네이버 증권 시가총액 순위 ─────────

KR_TAB = {1: "국내 시장지수", 2: "국내 업종·테마", 3: "국내 파생", 4: "해외 주식", 5: "원자재", 6: "채권", 7: "기타"}


CHINA_KR = re.compile(r"차이나|중국|항셍|홍콩|HSCEI|China|Hang\s?Seng|과창판|심천|CSI300|A50", re.I)


def kr_top(limit=None):
    url = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0&targetColumn=market_sum&sortOrder=desc"
    r = requests.get(url, headers=UA, timeout=30)
    r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "euc-kr"
    data = r.json()
    items = (data.get("result") or {}).get("etfItemList") or []
    print(f"한국 ETF 목록 받은 개수: {len(items)}")
    out = []
    for it in items:
        code = str(it.get("itemcode") or "")
        name = str(it.get("itemname") or "")
        tab = int(num(it.get("etfTabCode")) or 0) if str(it.get("etfTabCode") or "").strip() else 0
        if not re.fullmatch(r"[0-9A-Z]{6}", code) or tab == 3 or SKIP.search(name):
            continue
        out.append({"ticker": code + ".KS", "name": name, "cap": num(it.get("marketSum")),
                    "category": KR_TAB.get(tab, "국내 ETF")})
    out = [x for x in out if x["cap"] == x["cap"]]
    out.sort(key=lambda x: x["cap"], reverse=True)
    return out if limit == 0 else out[:KR_TOP]


def kr_china():
    """한국에 상장된 중국·홍콩 투자 ETF 전부 (레버리지·인버스 등은 SKIP 규칙으로 빠짐)"""
    return [x for x in kr_top(limit=0)
            if CHINA_KR.search(x["name"]) and not re.search(r"ex[\s-]?China", x["name"], re.I)]


# ───────── 미국 ETF: 운용자산 순위 ─────────

def us_top():
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            info = yf.Ticker(t).info or {}
            return t, float(info.get("totalAssets") or 0), str(info.get("longName") or info.get("shortName") or t), \
                str(info.get("category") or "")
        except Exception:
            return t, 0.0, t, ""

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = [x for x in ex.map(one, dict.fromkeys(US_CANDIDATES)) if x[1] > 0 and not SKIP.search(x[2])]
    res.sort(key=lambda x: x[1], reverse=True)
    print(f"미국 ETF 운용자산 받은 개수: {len(res)}")
    return [{"ticker": t, "name": n, "category": c or "미국 ETF"} for t, _, n, c in res[:US_TOP]]


def main():
    rows = load()
    have = {r["ticker"] for r in rows}
    added = []
    try:
        for i, x in enumerate(kr_top(), 1):
            if x["ticker"] in have:
                continue
            rows.append({"ticker": x["ticker"], "name": x["name"], "market": "KR",
                         "category": f"{x['category']} (시총 {i}위)", "theme": themes_for(x["name"])})
            have.add(x["ticker"])
            added.append(f"🇰🇷 {x['name']}")
    except Exception as e:
        print("한국 ETF 순위 업데이트 실패:", e)
    try:
        for x in kr_china():
            if x["ticker"] in have:
                continue
            where = "홍콩" if re.search(r"항셍|홍콩|HSCEI|차이나H|Hang\s?Seng", x["name"], re.I) else "중국"
            rows.append({"ticker": x["ticker"], "name": x["name"], "market": "KR",
                         "category": f"{where} 투자 (한국 상장)", "theme": themes_for(x["name"])})
            have.add(x["ticker"])
            added.append(f"🇰🇷→{where} {x['name']}")
    except Exception as e:
        print("한국 상장 중국·홍콩 ETF 추가 실패:", e)
    try:
        for x in us_top():
            if x["ticker"] in have:
                continue
            rows.append({"ticker": x["ticker"], "name": x["name"], "market": "US",
                         "category": x["category"], "theme": themes_for(x["name"] + " " + x["category"])})
            have.add(x["ticker"])
            added.append(f"🇺🇸 {x['name']}")
    except Exception as e:
        print("미국 ETF 순위 업데이트 실패:", e)
    save(rows)
    print(f"새로 넣은 ETF {len(added)}개 → 전체 {len(rows)}개")
    for a in added:
        print("  +", a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
