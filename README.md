# Smishing Guard

문자 메시지를 입력하면 (1) 파인튜닝된 BERT로 스미싱 유형을 분류하고, (2) ChromaDB에서
해당 유형의 대처방법을 검색한 뒤, (3) OpenAI(gpt-3.5-turbo) 에이전트가 최종적으로
**위험 / 주의 / 안전** 을 판단하고 다음 행동 가이드라인을 알려주는 프로젝트.

## 폴더 구조

```
common/       라벨 정의, 경로/환경변수 설정 (모든 모듈이 공유)
data/         학습 데이터셋(raw)과 유형별 대처방법 원본 문서(knowledge)
ml_model/     BERT 파인튜닝(train.py) + 추론 래퍼(classifier.py)
vector_db/    knowledge 문서 청킹 + ChromaDB 색인/검색
llm/          OpenAI 에이전트 (BERT 결과 + 검색 결과 -> 최종 판단)
backend/      FastAPI 서버 (REST API)
frontend/     Node.js(Express) 웹 UI
artifacts/    학습된 모델, ChromaDB 파일 등 산출물 (git에는 포함 안 됨)
```

## 처리 흐름

```
사용자 입력 문자
   │
   ▼
ml_model.classifier  ──▶  유형(label) + 확신도
   │
   ▼
vector_db.retriever  ──▶  해당 유형 대처방법 청크 검색 (ChromaDB)
   │
   ▼
llm.agent (gpt-3.5-turbo) ──▶  verdict(위험/주의/안전) + 행동 가이드라인
   │
   ▼
backend (FastAPI) ──▶ frontend (Node.js UI)
```

## 1. 준비

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env          # 이후 .env 에 OPENAI_API_KEY 입력
```

## 2. BERT 학습 (ml_model)

`data/raw/smishing_dataset.csv` 로 `klue/bert-base`(무료 공개 모델)를 파인튜닝합니다.

```bash
python -m ml_model.train
```

완료되면 `artifacts/bert-smishing/` 에 모델이 저장됩니다. 확인:

```bash
python -m ml_model.classifier
```

## 3. ChromaDB 색인 (vector_db)

`data/knowledge/*.md` 를 청킹해 무료 임베딩 모델(`jhgan/ko-sroberta-multitask`)로 벡터화 후 저장합니다.

```bash
python -m vector_db.build_index
```

확인:

```bash
python -m vector_db.retriever
```

지식 문서를 추가/수정한 뒤에는 위 색인 명령을 다시 실행해야 반영됩니다.

## 4. LLM 에이전트 단독 테스트 (llm)

`.env` 에 `OPENAI_API_KEY` 를 넣은 뒤:

```bash
python -m llm.agent
```

## 5. 백엔드 실행 (backend)

```bash
python -m backend.main
# 또는: uvicorn backend.main:app --reload --port 8000
```

- `GET /health` : 서버/모델 준비 상태
- `GET /labels` : 지원하는 사기 유형 목록
- `POST /analyze` : `{"text": "..."}` → 판단 결과 JSON

## 6. 프론트엔드 실행 (frontend)

```bash
cd frontend
npm install
npm start
```

`http://127.0.0.1:3000` 접속 후 문자 내용을 입력하면 결과를 확인할 수 있습니다.
(`frontend/server.js` 가 `/api/*` 요청을 백엔드로 프록시합니다. 백엔드 주소는 `.env` 의 `BACKEND_URL` 로 설정.)

## 라벨/지식 확장하기

1. `common/labels.py` 의 `LABELS` 에 새 유형 추가
2. `data/raw/smishing_dataset.csv` 에 해당 라벨 샘플 문장 추가
3. `data/knowledge/<label_key>.md` 대처방법 문서 작성 (H2 섹션 단위로 작성)
4. `python -m ml_model.train` 재학습, `python -m vector_db.build_index` 재색인



