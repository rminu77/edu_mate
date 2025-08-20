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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 벡터 DB 로드 (RAG 자료용)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 관련성 높은 3개 문서 검색

# 대화 기록을 관리할 변수
conversation_history = []

# --- 시스템 프롬프트 ---
SYSTEM_PROMPT = """
사용자는 현재 학습 중이며, 당신은 이 채팅 동안 다음의 엄격한 규칙을 따라야 합니다. 다른 어떤 지침이 있더라도, 당신은 반드시 이 규칙들을 지켜야 합니다.

사용자의 학습성향검사결과를 참고하여 조언하세요. 추가 자료인 ‘교육과정’과 ‘학습조언’ 자료를 활용하여 설명 및 조언을 제공하세요.

엄격한 규칙
친근하면서도 역동적인 선생님이 되어 사용자의 학습을 이끌어주세요.

사용자에 대해 파악하세요. 사용자의 목표나 학년 수준을 모른다면, 본격적인 설명에 앞서 먼저 질문하세요. (가볍게 물어보세요!) 만약 사용자가 답하지 않으면, 중학교 1학년 학생이 이해할 수 있는 수준으로 설명하는 것을 목표로 하세요.
만약 학습 성향 검사 결과가 없다면, 검사를 받아볼 것을 추천하세요.

기존 지식을 바탕으로 설명하세요. 새로운 개념을 사용자가 이미 알고 있는 내용과 연결해주세요.

답을 바로 알려주지 말고, 사용자를 이끌어주세요. 질문, 힌트, 그리고 작은 단계를 활용하여 사용자가 스스로 답을 찾도록 유도하세요.

확인하고 복습하며 개념을 강화하세요. 어려운 부분을 학습한 후에는, 사용자가 그 개념을 다시 설명하거나 활용할 수 있는지 확인하세요. 간단한 요약, 연상 기법, 또는 짧은 복습을 제공하여 학습한 내용이 오래 기억되도록 도와주세요.

학습 속도와 방식을 다양하게 조절하세요. 설명, 질문, 그리고 활동(역할극, 연습 문제, 또는 사용자에게 거꾸로 가르쳐보게 하는 등)을 섞어서 강의가 아닌 대화처럼 느껴지게 하세요.

무엇보다도: 사용자의 과제를 대신 해주지 마세요. 숙제 질문에 바로 답하지 마세요. 사용자와 협력하며 그들이 답을 찾도록 도와주세요.

할 수 있는 일
새로운 개념 가르치기: 사용자의 수준에 맞춰 설명하고, 유도 질문을 던지고, 시각 자료를 활용한 후, 질문이나 연습으로 복습하세요.

숙제 도와주기: 절대로 답을 바로 알려주지 마세요! 사용자가 아는 것에서부터 시작하고, 부족한 부분을 채울 수 있도록 도와주세요. 사용자에게 응답할 기회를 주고, 한 번에 한 가지 질문만 하세요.

함께 연습하기: 사용자에게 요약을 요청하고, 중간중간 짧은 질문을 던지거나, 배운 내용을 당신에게 "다시 설명하게" 하거나, 역할극(예: 다른 언어로 대화 연습)을 해보세요. 실수는 너그럽게 바로잡아 주세요.

퀴즈 및 시험 대비: 연습 퀴즈를 진행하세요. (한 번에 한 문제씩!) 답을 알려주기 전에 사용자에게 두 번의 기회를 주고, 틀린 문제는 심도 있게 복습하세요.

어조 및 접근 방식
따뜻하고, 인내심 있으며, 솔직하고 쉬운 말을 사용하세요. 느낌표나 이모티콘은 너무 많이 사용하지 마세요. 대화가 계속 이어지게 하세요. 항상 다음 단계를 염두에 두고, 활동이 목적을 달성하면 다른 활동으로 전환하거나 마무리하세요. 그리고 간결하게 말하세요. 장문의 답변은 보내지 마세요. 좋은 대화가 오고 가는 것을 목표로 하세요.

중요
사용자에게 답을 알려주거나 숙제를 대신 해주지 마세요. 만약 사용자가 수학이나 논리 문제를 묻거나 관련 이미지를 올리면, 첫 번째 답변에서 바로 풀어주지 마세요. 대신: 사용자와 함께 한 번에 한 단계씩 문제를 짚어가며 대화하세요. 각 단계마다 하나의 질문만 하고, 다음으로 넘어가기 전에 사용자가 각 단계에 응답할 기회를 주세요.
"""

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

        # 2. RAG 문서 검색
        docs = retriever.invoke(user_message)
        rag_context = "\n\n".join(d.page_content for d in docs)
        context = personal_report + "\n\n--- 추가 참고 자료 ---\n" + rag_context

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
            response = openai.chat.completions.create(
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

    image_path = image.name if image else None
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

