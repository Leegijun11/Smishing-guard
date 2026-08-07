# Smishing Guard

경량 ML 모델(KLUE-BERT)과 LLM(GPT-3.5)을 결합하여, 사용자가 입력한 문자 메시지의 스미싱 여부를 판단하고 유형별 맞춤형 대처 가이드를 제공하는 보안 보조 서비스입니다.

---

## 실행 화면

![Smishing Guard UI](./docs/demo_screen.PNG)

- **위험도 판별**: 분석 결과에 따른 위험 단계 (위험 / 주의 / 안전) 시각화
- **실시간 사기 유형 판별**: BERT 모델의 예측 확률 분포를 바 차트로 시각화
- **판단 근거 및 대처 가이드**: RAG로 검색된 공식 대처 문서 기반 가이드 제공

---

## Key Technical Decisions

### 1. Hybrid 2-Stage Pipeline (비용 및 속도 지연 최적화)
- **Problem**: 모든 문자 분석을 LLM API에 직결할 경우 처리 속도 지연과 API 비용 부담이 발생함.
- **Solution**: 
  - **1차 (Fast Path)**: Fine-tuned KLUE-BERT가 문자를 1차 분류 및 확률 분포 계산.
  - **2차 (Deep Analysis)**: BERT가 판별한 라벨 정보만 타겟팅하여 ChromaDB에서 대처 문서를 RAG 검색하고, 이를 기반으로 LLM이 최종 검증 및 맞춤형 행동 가이드 생성.
- **Result**: 불필요한 토큰 소모를 방지하고, 검색 노이즈를 최소화하여 정확도와 응답 속도를 동시 확보.

### 2. Single Source of Truth 기반 모듈화 설계
- **Problem**: 스미싱 유형이 추가/수정될 때 ML, Vector DB, LLM 프롬프트, API, UI 코드를 각각 수정하면서 발생하는 동기화 오류 위험.
- **Solution**: 
  - common/labels.py에 라벨 키, 한국어명, 프롬프트 설명, 룰베이스 키워드를 단일 공급원으로 관리.
  - 새 사기 유형 추가 시 labels.py 수정만으로 학습 데이터 로더, Vector DB 색인기, LLM 프롬프트 템플릿이 자동 동기화되는 Decoupled 구조 구축.

---

## 아키텍처 및 처리 흐름

1. **[사용자]** 문자 메시지 입력
2. **[ml_model]** KLUE-BERT 파인튜닝 모델
   - 1차 유형 분류 및 확률 분포 계산
3. **[vector_db]** ChromaDB Retriever
   - 분류된 라벨 메타데이터 기반 대처 문서 RAG 검색
4. **[llm]** OpenAI Agent (gpt-3.5-turbo)
   - 근거 기반 최종 verdict(위험/주의/안전) 및 행동 가이드 생성
5. **[backend/frontend]** FastAPI 및 Express Web UI를 거쳐 화면 출력

---

## 프로젝트 구조

- **common/**: [SSOT] 라벨 정의 및 공통 환경 설정 (전체 모듈 참조)
- **data/**: 학습용 raw 데이터셋 및 유형별 대처방법 지식 문서
- **ml_model/**: KLUE-BERT 파인튜닝(train.py) 및 추론 인터페이스(classifier.py)
- **vector_db/**: 지식 문서 청킹, 임베딩, ChromaDB 색인/검색 래퍼
- **llm/**: OpenAI 에이전트 및 프롬프트 템플릿
- **backend/**: FastAPI 기반 REST API 서버
- **frontend/**: Node.js Web UI
- **artifacts/**: 학습된 BERT 모델 및 ChromaDB 저장소 (Git 제외)

---

## Start

### 1. 환경 설정 및 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# .env 파일 생성 후 OPENAI_API_KEY 설정
```

### 2. BERT 모델 학습

`data/raw/smishing_dataset.csv` 데이터를 기반으로 `klue/bert-base`를 파인튜닝합니다.

```bash
python -m ml_model.train
```

학습 완료 후 추론 테스트:

```bash
python -m ml_model.classifier
```

### 3. ChromaDB 지식베이스 색인

`data/knowledge/*.md` 대처 문서를 청킹하여 임베딩(`jhgan/ko-sroberta-multitask`) 후 저장합니다.

```bash
python -m vector_db.build_index
```

### 4. 백엔드 및 프론트엔드 실행

#### 백엔드 실행 (FastAPI)

```bash
python -m backend.main
```

#### 프론트엔드 실행 (Node.js)

```bash
cd frontend
npm install
npm start
```

접속 주소

```
http://127.0.0.1:3000
```

---

## 사기 라벨 / 지식베이스 확장 방법

새로운 스미싱 유형을 추가하려면 아래 순서대로 진행합니다.

1. `common/labels.py`의 `LABELS`에 신규 라벨을 추가합니다.
2. `data/raw/smishing_dataset.csv`에 해당 라벨의 학습 데이터를 추가합니다.
3. `data/knowledge/<label_key>.md` 파일을 생성하고 대응 가이드를 작성합니다. (`##` 단위로 문서를 작성)
4. 모델과 지식베이스를 다시 생성합니다.

```bash
python -m ml_model.train
python -m vector_db.build_index
```
