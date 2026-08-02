from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LabelSpec:
    key: str            # 라벨
    ko: str             # 사용자에게 보여줄 한국어 이름
    description: str    # LLM 프롬프트에 넣을 유형 설명
    keywords: List[str] = field(default_factory=list)  # 룰베이스 폴백용


LABELS: List[LabelSpec] = [
    LabelSpec(
        key="delivery",
        ko="택배사칭",
        description=(
            "CJ대한통운·우체국·롯데택배 등 택배사나 배송기사를 사칭해 "
            "주소 오류, 배송 불가, 미수령 물품을 이유로 링크 클릭이나 앱 설치를 유도하는 유형."
        ),
        keywords=["택배", "송장", "운송장", "배송", "주소불명", "미배송", "물품보관", "우체국", "통관"],
    ),
    LabelSpec(
        key="institution",
        ko="공공기관·수사기관 사칭",
        description=(
            "검찰·경찰·금융감독원·국세청·건강보험공단·법원 등을 사칭해 "
            "수사 대상, 계좌 도용, 벌금·환급금 등을 언급하며 공포심을 유발하고 개인정보나 계좌 확인을 요구하는 유형."
        ),
        keywords=["검찰", "경찰", "금융감독원", "국세청", "법원", "출석", "수사", "영장", "환급금", "고발"],
    ),
    LabelSpec(
        key="financial",
        ko="금융·대출 사기",
        description=(
            "저금리 대환대출, 정부지원 서민대출, 카드 발급·한도 상향 등을 미끼로 "
            "선입금(보증금·신용등급 작업비)이나 개인·금융정보를 요구하는 유형."
        ),
        keywords=["대출", "대환", "저금리", "한도", "신용등급", "정부지원", "서민금융", "당일승인", "무담보"],
    ),
    LabelSpec(
        key="acquaintance",
        ko="지인·가족 사칭 (메신저피싱)",
        description=(
            "자녀·부모·친구를 사칭해 휴대폰 고장·액정 파손 등을 이유로 새 번호에서 연락하며 "
            "급하게 송금이나 상품권 구매, 원격제어 앱 설치를 요구하는 유형."
        ),
        keywords=["엄마", "아빠", "폰이", "액정", "지금 통화", "급하게", "부탁", "원격", "인증번호 좀"],
    ),
    LabelSpec(
        key="investment",
        ko="투자·리딩방 사기",
        description=(
            "주식 리딩방, 코인 선물, 해외 거래소, 원금보장 고수익을 앞세워 "
            "단톡방 초대나 비공식 거래 앱 설치, 특정 계좌 입금을 유도하는 유형."
        ),
        keywords=["리딩", "수익률", "원금보장", "코인", "선물", "단톡방", "종목추천", "상한가", "VIP방"],
    ),
    LabelSpec(
        key="payment",
        ko="결제·청구 사칭",
        description=(
            "해외 결제 승인, 소액결제 완료, 미납요금, 구독 자동결제 등 결제 사실을 허위로 통보하고 "
            "'본인 아닐 경우' 전화나 링크로 취소를 유도해 상담원 사칭으로 연결하는 유형."
        ),
        keywords=["결제", "승인", "소액결제", "미납", "요금", "취소", "고객센터", "환불", "청구"],
    ),
    LabelSpec(
        key="normal",
        ko="정상 문자",
        description="실제 기업·기관이 보내는 정상적인 안내 문자. 링크가 있어도 공식 도메인이며 금전·개인정보를 요구하지 않음.",
        keywords=[],
    ),
]

LABEL_KEYS: List[str] = [l.key for l in LABELS]
LABEL2ID: Dict[str, int] = {l.key: i for i, l in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: l.key for i, l in enumerate(LABELS)}
LABEL_MAP: Dict[str, LabelSpec] = {l.key: l for l in LABELS}

# 사기 인것들 리스트
SCAM_KEYS: List[str] = [l.key for l in LABELS if l.key != "normal"]


def ko_name(key: str) -> str:
    spec = LABEL_MAP.get(key)
    return spec.ko if spec else key


def describe(key: str) -> str:
    spec = LABEL_MAP.get(key)
    return spec.description if spec else ""


def label_catalog_text() -> str:
     
    lines = []
    for spec in LABELS:
        lines.append(f"- {spec.key} ({spec.ko}): {spec.description}")
    return "\n".join(lines)
