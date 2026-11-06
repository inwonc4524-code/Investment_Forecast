# 📊 ETF 추천 봇

매일 아침(한국시간 화~토 오전 7시쯤) 한국·미국 ETF 29개를 분석해서
**휴대폰으로 요약 알림**을 보내고, **웹 대시보드**에 전체 순위를 올려줍니다.

## 무엇을 기준으로 골라주나요?

| 카드 | 기준 |
|---|---|
| 🏆 종합 TOP 5 | 상승 흐름 점수 60% + 안정성 점수 40%, 과열(RSI 75 이상)이면 감점 |
| 📈 상승 흐름 | 1·3·6개월 수익률이 좋고, 200일 평균선 위에 있는 ETF |
| 🔄 반등 후보 | 200일 평균선은 오르는 중인데, 최근 RSI가 40 미만이거나 1년 고점보다 8% 넘게 빠진 ETF |
| 🛡 안정형 | 변동성·최대낙폭이 작고, 6개월 수익률이 플러스인 ETF |

"과열" 표시는 RSI 70 이상, 즉 최근 단기간에 많이 올라 숨 고르기가 나올 수 있다는 뜻이에요.

> ⚠ 과거 가격만으로 계산한 참고 지표예요. 앞으로의 수익을 보장하지 않으며 투자 권유가 아닙니다.

---

## 설치 방법 (코딩 필요 없음, 약 10분)

### 1. GitHub에 새 저장소 만들기
1. github.com 접속 → 오른쪽 위 **+** → **New repository**
2. 이름: `etf-bot` (아무거나 괜찮아요) → **Public** 선택 → **Create repository**
   (GitHub Pages 무료 대시보드는 Public 저장소에서 쓸 수 있어요. ETF 가격 정보뿐이라 공개돼도 괜찮습니다.)

### 2. 파일 올리기
1. 저장소 화면에서 **uploading an existing file** 클릭
2. 받은 zip을 푼 폴더 안의 파일을 **전부** 끌어다 놓기
   (`.github` 폴더도 꼭 포함! 안 보이면 맥은 Finder에서 `Cmd+Shift+.` 로 숨김 파일을 보이게 하세요)
3. **Commit changes**

### 3. ntfy 알림 연결 (구인 알림 봇과 같은 방식)
1. 휴대폰 ntfy 앱 → **+** → 남들이 못 맞힐 주제 이름 입력 (예: `inwon-etf-7k3q`) → 구독
2. GitHub 저장소 → **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `NTFY_TOPIC`
   - Secret: 방금 만든 주제 이름 (예: `inwon-etf-7k3q`)

### 4. 봇이 결과를 저장할 수 있게 허용
**Settings → Actions → General** → 맨 아래 **Workflow permissions** →
**Read and write permissions** 선택 → **Save**

### 5. 첫 실행
**Actions** 탭 → 왼쪽 **ETF 추천 봇** → **Run workflow** → 1~2분 뒤 휴대폰에 알림 도착 🎉

### 6. 대시보드 켜기 (첫 실행 후에)
**Settings → Pages** → Source: **Deploy from a branch** → Branch: `main`, 폴더: `/docs` → **Save**
1~2분 뒤 `https://내아이디.github.io/etf-bot/` 에서 대시보드가 열려요.
알림의 **대시보드 열기** 버튼을 누르면 바로 이동합니다.

---

## 종목 바꾸기
`etfs.csv` 파일을 GitHub에서 열고 연필(✏️) 버튼으로 수정하면 돼요.

- 한국 ETF: 종목코드 뒤에 `.KS` (예: `069500.KS`)
- 미국 ETF: 티커 그대로 (예: `SPY`)
- 상장한 지 1년이 안 된 ETF는 계산에서 빠져요.

## 알림 시간 바꾸기
`.github/workflows/etf-bot.yml` 의 `cron: "13 22 * * 1-5"` 부분이에요. 시간은 한국시간에서 9시간을 뺀 값(UTC)으로 적습니다.
원하는 시간을 Claude에게 말해주면 바꿔드릴게요.
