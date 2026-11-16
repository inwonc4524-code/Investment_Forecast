"""봇이 만든 알림을 '새 화면이 인터넷에 올라간 뒤'에 보내요.

봇은 알림을 바로 보내지 않고 .pending_alerts.json 에 적어 둬요(PENDING_ALERTS 환경변수).
대시보드를 저장(push)한 뒤 이 파일이 GitHub Pages 의 version.json 이 새 번호로 바뀔 때까지
10초마다 확인하고(최대 6분), 바뀌면 알림을 보내요. 그래야 알림을 누르자마자 새 화면이 열려요.
6분이 지나도 안 바뀌면(Pages가 늦을 때) 알림이 안 가는 것보다 낫기 때문에 그냥 보내요.
사용법: python send_later.py stock_bot   (또는 etf_bot)
"""
import importlib
import json
import os
import sys
import time

import requests

PENDING = ".pending_alerts.json"


def main():
    mod = sys.argv[1] if len(sys.argv) > 1 else "stock_bot"
    if not os.path.exists(PENDING):
        print("보낼 알림 없음")
        return
    alerts = json.load(open(PENDING, encoding="utf-8"))
    if not alerts:
        print("보낼 알림 없음")
        return
    build = json.load(open("docs/version.json", encoding="utf-8")).get("build")
    owner, name = os.environ.get("GITHUB_REPOSITORY", "/").split("/")
    live = f"https://{owner}.github.io/{name}/version.json"
    t0 = time.time()
    while time.time() - t0 < 360:
        try:
            v = requests.get(f"{live}?t={int(time.time())}", timeout=10,
                             headers={"Cache-Control": "no-cache"}).json().get("build")
            if v and v >= build:
                print(f"새 화면 올라감 ({v}) — {time.time() - t0:.0f}초 기다림")
                break
        except Exception as e:
            print("화면 확인 실패(다시 시도):", e)
        time.sleep(10)
    else:
        print("6분 안에 새 화면이 안 보여서 그냥 보냄")
    os.environ.pop("PENDING_ALERTS", None)
    bot = importlib.import_module(mod)
    for a in alerts:
        bot.send_ntfy(a["title"], a["message"], a["click_url"])
        print("알림 보냄:", a["title"])
    os.remove(PENDING)


if __name__ == "__main__":
    main()
