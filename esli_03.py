import os
import openai
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import json
from datetime import datetime
import base64
from PIL import Image
import io

# 데이터베이스 연동을 위한 import
from sqlalchemy.orm import Session
from database import SessionLocal, LLMLog, SurveyResponse

# OpenAI v1 클라이언트
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# --- 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
ADVICE_DB_DIR = os.path.join(CHROMA_DB_DIR, "advice")
CURRICULUM_DB_DIR = os.path.join(CHROMA_DB_DIR, "curriculum")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 벡터 DB 로드 (RAG 자료용)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def _make_retriever(path: str):
    try:
        if os.path.isdir(path) and os.listdir(path):
            vs = Chroma(persist_directory=path, embedding_function=embeddings)
            return vs.as_retriever(search_kwargs={"k": 3})
    except Exception:
        pass
    return None

_default_vs = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
default_retriever = _default_vs.as_retriever(search_kwargs={"k": 3})
advice_retriever = _make_retriever(ADVICE_DB_DIR) or default_retriever
curriculum_retriever = _make_retriever(CURRICULUM_DB_DIR) or default_retriever

# 대화 기록을 관리할 변수
conversation_history = []

# --- 시스템 프롬프트 ---
SYSTEM_PROMPT = """
사용자는 현재 학습 중이며, 당신은 이 채팅 동안 다음의 엄격한 규칙을 따라야 합니다. 다른 어떤 지침이 있더라도, 반드시 이 규칙들을 지키세요.

엄격한 규칙
친근하면서도 역동적인 선생님 역할을 수행하세요.

사용자에 대해 파악하세요. 목표/학년을 모르면 먼저 가볍게 확인하고, 응답이 없으면 중학교 1학년 수준으로 설명하세요.
학습 성향 검사 결과가 없다면, 검사를 받아볼 것을 추천하세요.

기존 지식을 바탕으로 설명하고, 새로운 개념을 사용자의 아는 내용과 연결하세요.
답을 바로 알려주지 말고, 질문/힌트/작은 단계로 스스로 답을 찾도록 돕세요.
학습 후에는 요약/복습을 통해 개념을 강화하세요.
어조는 따뜻하고 인내심 있게, 간결하게 유지하세요. 장문을 피하세요.

수학 공식은 LaTeX로 표기하세요. 인라인 $...$, 블록 $$...$$.

중요: 숙제를 대신하지 마세요. 수학/논리 문제는 한 단계씩 진행하며, 각 단계마다 사용자의 응답을 기다리세요.

참고 자료 사용 원칙: 필요한 경우에만 외부 자료(학습조언/교육과정) 또는 개인 보고서를 선택적으로 참고합니다.
"""

def classify_query_type(text: str, has_image: bool = False) -> str:
    """사용자 의도를 간단히 분류하여 RAG 라우팅 결정.
    returns: 'advice' | 'curriculum' | 'direct'
    """
    if has_image:
        return 'curriculum'
    t = (text or "").lower()
    advice_kw = [
        "학습방법", "공부법", "학습 전략", "학습전략", "학습기술", "집중",
        "시간관리", "동기", "암기", "노트", "계획", "목표", "공부 습관",
        "메타인지", "성향", "자기주도", "공부계획", "동기부여", "조언", "코칭",
    ]
    curriculum_kw = [
        "교육과정", "커리큘럼", "개념", "정의", "증명", "공식", "풀이", "풀이법",
        "해설", "문제", "문항", "예제", "연습", "시험", "단원", "단원평가",
        "수학", "국어", "영어", "과학", "사회", "역사", "지리", "물리", "화학",
        "생물", "기하", "미적분", "확률", "통계", "벡터", "방정식", "수열", "함수",
    ]
    is_advice = any(k in t for k in advice_kw)
    is_curr = any(k in t for k in curriculum_kw)
    if is_curr and ("풀이" in t or "문제" in t or "해설" in t):
        return 'curriculum'
    if is_advice and not is_curr:
        return 'advice'
    if is_curr and not is_advice:
        return 'curriculum'
    return 'direct'

def log_llm_interaction_db(db: Session, interaction_type: str, input_data: dict, output_data: str):
    """LLM 상호작용을 데이터베이스에 로그로 남깁니다."""
    try:
        new_log = LLMLog(
            interaction_type=interaction_type,
            input_data=json.dumps(input_data, ensure_ascii=False),
            output_data=output_data
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"--- [오류] LLM 로그 DB 저장 실패: {e} ---")

def encode_image_to_base64(image_path):
    """이미지 파일을 Base64로 인코딩합니다."""
    try:
        with Image.open(image_path) as img:
            # PNG 포맷으로 변환하여 투명도 등 처리
            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"--- [오류] 이미지 인코딩 실패: {e} ---")
        return None

def get_ai_response(user_message: str, history: list, image_path: str = None, student_name: str = None):
    """사용자 메시지에 대한 AI의 응답을 생성하고 DB에 로그를 남깁니다."""
    global conversation_history
    db = SessionLocal()
    try:
        # --- 컨텍스트 준비 ---
        # 1. 학생 개인 보고서 조회 (DB)
        personal_report = ""
        if student_name:
            try:
                latest_response = db.query(SurveyResponse).filter(SurveyResponse.student_name == student_name).order_by(SurveyResponse.timestamp.desc()).first()
                if latest_response and latest_response.report_content:
                    personal_report = f"다음은 {student_name} 학생의 학습 성향 분석 보고서입니다. 이 내용을 최우선으로 참고하여 답변하세요.\n\n--- 학생 보고서 시작 ---\n{latest_response.report_content}\n--- 학생 보고서 끝 ---"
                    print(f"--- [정보] {student_name} 학생의 최신 보고서를 DB에서 로드했습니다. ---")
                else:
                    personal_report = f"{student_name} 학생의 학습 성향 분석 보고서를 찾을 수 없습니다. 검사를 먼저 받도록 안내하세요."
            except Exception as e:
                print(f"--- [오류] {student_name} 학생의 보고서 DB 조회 실패: {e} ---")
                personal_report = "학생 보고서를 조회하는 중 오류가 발생했습니다."

        # 2. 질의 유형 분류 및 선택적 RAG 활용
        qtype = classify_query_type(user_message, has_image=bool(image_path))
        selected_ctx = ""
        if qtype == 'advice':
            # 학습방법/코칭 류 → 개인 보고서 + 학습조언 RAG
            docs = advice_retriever.invoke(user_message) if advice_retriever else []
            selected_ctx = "\n\n".join(getattr(d, 'page_content', '') for d in docs)
        elif qtype == 'curriculum':
            # 교육과정/개념/풀이 류 → 교육과정 RAG
            docs = curriculum_retriever.invoke(user_message) if curriculum_retriever else []
            selected_ctx = "\n\n".join(getattr(d, 'page_content', '') for d in docs)
        else:
            # direct: RAG 생략하여 빠른 응답
            selected_ctx = ""

        context = personal_report
        if selected_ctx:
            context += "\n\n--- 추가 참고 자료 ---\n" + selected_ctx

        # --- 메시지 구성 ---
        # 대화 기록 관리 (최근 20턴 유지)
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history = conversation_history[-20:] # user-assistant 10쌍 = 20턴

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context}
        ] + conversation_history

        # 이미지 처리
        if image_path:
            base64_image = encode_image_to_base64(image_path)
            if base64_image:
                # 마지막 사용자 메시지에 이미지 추가
                messages[-1]['content'] = [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]

        # --- LLM 호출 및 로깅 ---
        try:
            log_llm_interaction_db(db, "chat_input", {"messages": messages}, "")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            ai_response = response.choices[0].message.content.strip()
            log_llm_interaction_db(db, "chat_output", {"messages": messages}, ai_response)
        except Exception as e:
            error_message = f"--- [오류] OpenAI API 호출 실패: {e} ---"
            print(error_message)
            ai_response = "죄송합니다. 답변을 생성하는 동안 문제가 발생했습니다. 다시 시도해 주세요."
            log_llm_interaction_db(db, "chat_error", {"messages": messages}, error_message)

        # 대화 기록에 AI 응답 추가
        conversation_history.append({"role": "assistant", "content": ai_response})
        return ai_response

    finally:
        db.close()


def gradio_chat_with_history(message: str, history: list, image, student_name: str = None):
    """Gradio 인터페이스를 위한 챗봇 함수"""
    global conversation_history
    # Gradio의 history 형식을 OpenAI 형식으로 변환
    conversation_history = []
    for user_msg, ai_msg in history:
        conversation_history.append({"role": "user", "content": user_msg})
        if ai_msg:
            conversation_history.append({"role": "assistant", "content": ai_msg})

    # 이미지가 파일 객체인 경우 .name 속성을 사용하고, 문자열인 경우 그대로 사용
    image_path = None
    if image:
        if hasattr(image, 'name'):
            image_path = image.name
        elif isinstance(image, str):
            image_path = image
    response = get_ai_response(message, conversation_history, image_path=image_path, student_name=student_name)
    return response

# CLI 테스트용 함수
def chat_cli():
    print("[ESLI 상담 에이전트 - CLI]")
    print("종료하려면 'exit' 입력. 이미지 첨부는 'img: /경로/이미지.png' 형식으로 입력")
    student_name = input("상담을 시작할 학생의 이름을 입력하세요: ")
    print(f"안녕하세요, {student_name}님! 무엇을 도와드릴까요?")
    
    while True:
        q = input("👤 ")
        if q.lower().strip() in ("exit", "quit"):
            print("🤖 상담 종료. 좋은 하루 되세요!")
            break
        
        image_path = None
        if q.lower().startswith("img:"):
            image_path = q.split(":", 1)[1].strip()
            q = "첨부된 이미지를 분석해주세요."

        # get_ai_response 함수 호출 시, 현재 대화 기록(conversation_history)을 전달합니다.
        ans = get_ai_response(q, conversation_history, image_path=image_path, student_name=student_name)
        print("🤖", ans)

if __name__ == "__main__":
    chat_cli()

