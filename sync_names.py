"""
목록(csv)의 이름을 네이버 증권에 등록된 공식 이름으로 맞춰요. (매달 목록 업데이트 때 자동 실행)
  🇰🇷 한국 주식·ETF: 네이버 증권
  🇺🇸 미국 · 🇯🇵 일본 · 🇭🇰 홍콩 · 🇨🇳 중국 · 🇻🇳 베트남: 네이버 증권 해외주식 (한글 이름)
  다른 나라는 그대로 둬요.
회사 이름이 바뀌었거나(예: 사명 변경), 처음에 잘못 적힌 이름을 바로잡아요.
실행: python sync_names.py <csv 파일>
"""
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from crosscheck import UA, WORLD, naver_world_codes


def naver_name(ticker, mk):
    try:
        if mk == "KR":
            code = ticker.split(".")[0]
            r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/basic", headers=UA, timeout=12)
            if r.ok:
                return (r.json().get("stockName") or "").strip() or None
        elif mk in WORLD:
            for code in naver_world_codes(ticker, mk):
                r = requests.get(f"https://api.stock.naver.com/stock/{code}/basic", headers=UA, timeout=12)
                if r.ok and r.json().get("stockName"):
                    return r.json()["stockName"].strip()
    except Exception:
        return None
    return None


def main(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    def one(r):
        time.sleep(0.1)
        return r["ticker"], naver_name(r["ticker"], r["market"])

    with ThreadPoolExecutor(max_workers=4) as ex:
        found = dict(ex.map(one, rows))
    changed = []
    for r in rows:
        new = found.get(r["ticker"])
        if new and new != r["name"]:
            changed.append(f"{r['ticker']}: {r['name']} → {new}")
            r["name"] = new
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"네이버 이름 확인 {sum(1 for v in found.values() if v)}/{len(rows)}곳 · 이름을 고친 곳 {len(changed)}곳")
    for c in changed:
        print("  ", c)


if __name__ == "__main__":
    main(sys.argv[1])
