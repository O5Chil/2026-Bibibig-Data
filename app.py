from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


DB_DIR = "db"
TARGET_MAJOR = "의공학과, 바이오메디컬공학과, 의료공학과, 생체의공학과"


print("의공학 면접관 AI를 불러오는 중입니다...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2
)


def shorten_text(text, max_length=420):
    text = text.replace("\n", " ").strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def search_reference(question, student_answer):
    search_query = f"""
    {TARGET_MAJOR} 대학 면접 평가 기준
    의공학 바이오메디컬공학 의료기기 생체신호 생명과학 공학적 문제해결
    전공적합성 지원동기 학업계획 진로계획 융합적 사고 과학탐구능력 논리적 표현력

    면접 질문:
    {question}

    학생 답변:
    {student_answer}
    """

    docs = db.similarity_search(search_query, k=5)

    evidence_blocks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "출처 없음")
        source_name = Path(source).name

        page = doc.metadata.get("page", None)
        if isinstance(page, int):
            page_text = f"{page + 1}쪽"
        else:
            page_text = "페이지 정보 없음"

        text = shorten_text(doc.page_content, max_length=420)

        evidence_blocks.append({
            "id": f"E{i}",
            "source": source_name,
            "page": page_text,
            "text": text
        })

    return evidence_blocks


def make_evidence_context(evidence_blocks):
    context = ""

    for ev in evidence_blocks:
        context += f"\n[{ev['id']}]\n"
        context += f"근거 원문: {ev['text']}\n"
        context += f"출처: {ev['source']}, {ev['page']}\n"

    return context


def evaluate_answer(question, student_answer):
    evidence_blocks = search_reference(question, student_answer)
    evidence_context = make_evidence_context(evidence_blocks)

    prompt = f"""
너는 {TARGET_MAJOR} 대학 입학 면접관 AI이다.

중요 규칙:
- 반드시 의공학, 바이오메디컬공학, 의료기기, 생체신호, 생명과학, 공학적 문제해결, 의료와 공학의 융합 관점에서 평가한다.
- 컴퓨터공학, 물리치료, 간호학, 의예과, 약학, 사범대, 체육, 유아교육 등 다른 전공 이야기로 벗어나지 않는다.
- 단, 학생 답변에 프로그래밍, 인공지능, 센서, 데이터 분석, 3D 프린팅, 로봇, 전자회로 등이 나오면 의공학과 연결해서 평가한다.
- 참고자료에 다른 전공 내용이 섞여 있어도 의공학과 직접 관련 없는 내용은 무시한다.
- 참고자료에 없는 내용을 확정적으로 말하지 않는다.
- 학생을 비난하지 말고 실제 면접 피드백처럼 구체적으로 조언한다.
- 내부 추론 과정은 출력하지 않는다.

근거 사용 규칙:
- 아래 [검색된 PDF 근거]에 있는 내용만 근거로 사용한다.
- 근거 원문과 출처는 제공된 내용을 그대로 사용한다.
- PDF 파일명이나 페이지를 새로 만들지 않는다.
- "면접 질문 주제", "다음 질문 주제", "평가 기준"을 출처로 쓰지 않는다.
- 근거가 약하면 "직접적인 근거는 아니지만"이라고 표현한다.
- 근거가 전혀 없으면 "(근거 부족)"이라고 쓴다.
- 각 평가 문장 바로 아래에 반드시 아래 형식으로 붙인다.
- 절대로 빈 항목을 만들지 마라.
- 평가 근거, 좋았던 점, 아쉬운 점, 개선 방향, 모범적인 답변 방향에는 각각 최소 1개 이상의 내용을 작성하라.

형식:
  (근거: "PDF 원문 일부" / 출처: PDF 파일명, 페이지)

[검색된 PDF 근거]
{evidence_context}

[면접 질문]
{question}

[학생 답변]
{student_answer}

평가 기준:
1. 전공적합성
2. 지원동기의 구체성
3. 과학·공학적 사고력
4. 문제 해결 능력
5. 생명과학과 공학의 융합 이해도
6. 논리적 표현력
7. 발전 가능성

다음 질문은 반드시 아래 주제 중 하나로 만들어라:
- 의공학과에 지원한 이유
- 의료기기나 바이오 기술에 관심을 갖게 된 계기
- 생명과학과 공학을 연결해 본 경험
- 의료 문제를 공학적으로 해결하고 싶은 사례
- 센서, 인공지능, 데이터 분석, 프로그래밍을 의료 분야에 활용하는 방법
- 고등학교 과학 탐구나 프로젝트 경험
- 의공학자가 갖추어야 할 윤리의식
- 앞으로 개발하고 싶은 의료기기 또는 헬스케어 기술

아래 형식으로만 답변하라.

평가 점수: 10점 만점 중 __점

평가 신뢰도: 높음 / 보통 / 낮음

평가 근거:
- 학생 답변이 면접 평가 기준에 비추어 어떤 수준인지 구체적으로 평가한다.
  (근거: "PDF 원문 일부" / 출처: PDF 파일명, 페이지)

좋았던 점:
- 학생 답변에서 긍정적으로 볼 수 있는 점을 작성한다.
  (근거: "PDF 원문 일부" / 출처: PDF 파일명, 페이지)

아쉬운 점:
- 학생 답변에서 부족한 점을 작성한다.
  (근거: "PDF 원문 일부" / 출처: PDF 파일명, 페이지)

개선 방향:
- 답변을 더 좋게 만들기 위해 추가하면 좋은 내용을 작성한다.
  (근거: "PDF 원문 일부" / 출처: PDF 파일명, 페이지)

모범적인 답변 방향:
- 더 좋은 답변이 되려면 어떤 식으로 말하면 좋은지 예시 방향을 작성한다.
  (근거: "PDF 원문 일부" / 출처: PDF 파일명, 페이지)

주의할 점:
- 이 평가는 실제 대학 면접 점수를 완전히 대체할 수 없으며, PDF 자료 기반의 면접 연습용 피드백이다.

다음 질문:
"""

    response = llm.invoke(prompt)
    return response.content


def extract_next_question(result):
    lines = result.splitlines()

    for line in lines:
        if line.strip().startswith("다음 질문"):
            next_q = line.replace("다음 질문:", "").strip()
            if next_q:
                return next_q

    return "의공학과에 지원하게 된 계기와, 의료 문제를 공학적으로 해결하고 싶다고 느낀 경험을 설명해 주세요."


print("\n===================================")
print("의공학 대학 면접관 AI")
print("종료하려면 q 또는 quit 입력")
print("===================================\n")

question = "자기소개와 의공학과 지원동기를 말해 주세요."

while True:
    print(f"\n면접관 질문: {question}")
    student_answer = input("\n학생 답변: ")

    if student_answer.lower() in ["q", "quit", "exit", "종료"]:
        print("면접을 종료합니다.")
        break

    print("\n답변 평가 중...\n")

    result = evaluate_answer(question, student_answer)

    print("===================================")
    print(result)
    print("===================================")

    question = extract_next_question(result)