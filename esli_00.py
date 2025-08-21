import gradio as gr
from datetime import datetime
from typing import Dict, List
import pandas as pd
import os
import random
import json
import uuid

# --- 프로젝트 모듈 임포트 ---
from esli_01 import calculate_scores
from esli_02 import generate_report_with_llm
from esli_03 import gradio_chat_with_history
from database import SessionLocal, SurveyProgress, init_db

# --- 질문 목록 정의 ---
# (기존 questions_part1, questions_part2, questions_part3 변수 내용은 여기에 그대로 유지됩니다)
# Part I: 학업관련 감정과 행동 패턴
questions_part1 = {
    "감정과 행동 패턴 (1/7)": [
        "마음대로 일이 되지 않으면 불안하다", "다른 친구들에 비해 머리가 많이 나쁜것 같다", "학교에서 은근 따돌림 받는 것 같다",
        "나에 관한 결정에서 부모님은 항상 내 의견 묻고 결정한다", "학교에서 배우는 것이 나에게 많은 도움이 될 것 같다",
        "좋지 않는 생각이 떠오르면 자꾸 그 생각이 나서 기분이 나빠진다", "내가 공부한 만큼 또는 그 이상 결과를 얻고 있다",
        "학교에서 먼저 반겨주고 이야기 건네주는 친구가 있다", "부모님과 밥 먹는 것이 힘들다", "학교 선생님에게 꾸중보다 칭찬을 더 듣는다"
    ],
    "감정과 행동 패턴 (2/7)": [
        "좋지 않은 일이 생기면 배가 아픈 경우가 많다", "아무리 노력해도 좋은 성적을 받을 수 없을 것 같다", "학교 끝나고 친구들과 만나서 놀기도 한다",
        "부모님이 형제자매나 사촌과 비교하는 말을 자주한다", "담임선생님이 내가 나쁜 학생이라고 불친절하다", "기분 나쁜 일을 겪어도 곧 기분을 푼다",
        "머리가 좋은 편이어서 친구들보다 쉽게 공부한다", "친구들에게 내가 먼저 말을 걸기가 어렵다", "모든것을 마음대로 결정하는 부모님 때문에 답답하다",
        "방학때 너무 심심해서 차라리 학교가고 싶다는 생각을 한 적이 있다"
    ],
    "감정과 행동 패턴 (3/7)": [
        "작은 일도 다른 사람들이 어떻게 생각할지 신경이 많이 쓰인다", "어려운 문제라도 충분히 시간이 주어지면 스스로 풀어낼 자신이 있다",
        "주변 친구들은 대부분 나와 사이가 좋지 않다", "갑자기 깜짝 놀랄 정도로 부모님께서 화를 내는 경우가 자주 있다",
        "학교에 가면 나도 모르게 무섭거나 짜증나서 별로 가고 싶지 않다", "하고 싶어하는 일을 하기도 전에 잘못될 걱정을 많이 한다",
        "혼자 문제를 푸는 것보다 다른 사람에게 물어보는 것이 낫다", "친구들과 함께 노는 일은 정말 즐겁다",
        "가족 중에 나를 이해해주는 사람이 있어서 고민을 이야기 할 수 있다", "학교에 가는 것은 재미있는 일이다"
    ],
    "감정과 행동 패턴 (4/7)": [
        "한 번도 다른 사람에게 거짓말을 해본 적이 없다", "부탁하는 사람들 때문에 가끔 짜증이 날 때도 있다", "실수했을 때 항상 다른 사람에게 사과하고 인정한다",
        "마음에 들지 않는 사람에게도 언제나 예의바르게 행동한다", "어른이 하는 말이 맞다는 것을 알면서도 반항하고 싶었던 적이 있다",
        "능력이 부족하다고 생각해서 어떤 일을 중간에 그만둔 적이 있다", "어느 누구와 이야기해도 다른 사람 말을 잘 들어준다",
        "마음대로 하지 못하면 화가 날 때도 있다", "다른 사람을 이용해서 이익을 얻으려 한 적이 한 번도 없다", "다른 친구가 잘 되는 것이 부러웠던 적이 있다"
    ],
    "감정과 행동 패턴 (5/7)": [
        "억울한 일을 당했을 때, 복수하려는 생각을 가져본 적이 있다", "혼자 공부하다 잠이 오면 잠을 깨는 나만의 방법이 있다",
        "공부시작하면 마칠 때까지 거의 공부만 한다", "숙제를 하다가도 꼭 봐야하는 TV프로그램이 있다",
        "매일 1시간 이상 공부와 관련 없이 컴퓨터를 한다 (게임, 인터넷 등)", "핸드폰이나 스마트기기가 없어도 내 생활에 큰 영향은 없다",
        "밤10시만 되도 잠이 와서 공부하기 어렵다", "공부하려고 앉으면 10분도 안되서 딴 생각에 빠진다", "하루에 1~2시간 이상 TV를 본다",
        "공부하다가도 게임 인터넷 생각이 나면 컴퓨터를 해야 마음이 편하다"
    ],
    "감정과 행동 패턴 (6/7)": [
        "문자나 인터넷, 게임 등을 위해 하루 1시간 이상 핸드폰을 한다", "학교 수업시간에 자는 경우가 많다",
        "제대로 공부에 집중하려면 최소 10분이상 준비할 시간이 필요하다", "TV드라마나 어린이 프로그램 한 두편 정도 보는 것은 크게 상관없다",
        "한밤에 가족들이 모두 자는 동안 몰래 컴퓨터를 하는 경우가 많다", "핸드폰이 거의 1분 간격으로 카톡 알림이 울린다",
        "단원평가나 학교시험을 위해서는 평소보다 잠을 줄여 공부하는 편이다", "선생님이나 교과서의 설명이 무슨 말인지 알아들을 수가 없다",
        "친구들과 대화하는 거의 모든 주제는 TV프로그램과 관련된 것이다", "단원평가 전이나 시험 준비할 때에는 게임이나 인터넷에 접속하지 않는다"
    ],
    "감정과 행동 패턴 (7/7)": [
        "친구들과 톡을 주고 받지 못하면 불안하다", "하루 평균 10시간 이상 자는 것 같다", "공부하다가 나도 모르게 시간이 훌쩍 지나간 경우가 많다",
        "내가 좋아하는 TV프로그램을 놓치면 궁금해서 다른 일을 할 수가 없다", "학교나 학원 친구보다 게임, 인터넷 커뮤니티 친구들과 더 친하다",
        "스마트폰 데이터가 다 떨어져서 사용못하면 매우 답답하다"
    ]
}

# Part II: 학습 방법 및 기술
questions_part2 = {
    "학습 방법 및 기술 (1/7)": [
        "어떤 일을 하기 전에 항상 목표세워 시작한다", "공부 계획을 위한 다이어리나 계획표를 사용한다", "공부하다보면 쉽게 피곤해져서 계속 공부하기 어렵다",
        "단원평가나 학원 테스트 끝나면 틀린 문제를 다시 풀며 틀린 이유를 확인한다", "공부할 때는 학습 목표를 꼭 확인한다",
        "공부할 내용의 뜻을 이해하기 보다는 바로 외우는 편이다", "과목에 따라 다르게 사용하는 정리 노트들을 가지고 있다",
        "중요한 외울 내용들은 꼭 다 외우면서 공부한다", "단원평가를 보면 거의 생각했던 문제가 출제된다"
    ],
    "학습 방법 및 기술 (2/7)": [
        "특별히 되고 싶은 직업이나 장래 희망이 없다", "따로 공부계획을 세우지 않고 그날 그날 공부한다", "공부하기로 마음 먹고 나서도 한참 지나야 겨우 공부를 시작한다",
        "하루 마무리 할 때에는 오늘 했었던 일을 정리하는 시간을 갖는다", "공부하다가 잘 모르는 단어가 나오면 무슨 뜻인지 찾아보고 넘어간다",
        "잘 이해되지 않는 내용은 어떻게든 꼭 알아보고 넘어가야 마음이 놓인다", "공부한 내용을 정리노트나 마인드맵을 이용해 공부하지 않는다",
        "참고서에 잘 정리되어 있어서, 굳이 내가 직접 공부한 내용을 정리할 필요는 없다", "처음 보는 문제도 당황하지 않고 풀어서 맞출 수 있다"
    ],
    "학습 방법 및 기술 (3/7)": [
        "미래에 성공한 나의 모습을 상상하면 마음이 설렌다", "계획을 세워도 지키지 않는 경우가 더 많다", "TV나 주변 소리에도 크게 신경쓰지 않고 공부할 수 있다",
        "문제를 푼 뒤에는 몇 개를 맞고 틀렸는지만 확인하고 넘어간다", "다른 사람의 도움 없이는 새로 배우는 단원의 내용을 이해하기가 어렵다",
        "새로운 내용을 배우면 전에 배운 내용과 비교하면서 공부한다", "수업을 들으면 전에 배운 내용들과 관계를 연결지어 공부할 수 있다",
        "암기할 때 주로 사용하는 나만의 암기법이 있다", "제시된 문제를 잘못 읽어 틀린 문제가 자주 발견된다"
    ],
    "학습 방법 및 기술 (4/7)": [
        "누군가가 목표를 정해주고 나는 그대로 시키는대로만 했으면 좋겠다", "하루에 얼만큼 공부할 수 있을지 잘 모르기 때문에 계획을 미리 짜는 것은 불가능하다",
        "오늘 할 일은 미루지 않고 끝내려고 노력한다", "좋지 않는 결과가 나오면 왜 그렇게 되었는지 생각해보는 편이다",
        "수업시간에 수업 듣지 않고 자거나 다른 숙제를 하는 편이다", "혼자서 공부하는 것 보다는 남이 가르쳐주는 것을 듣는 것이 훨씬 좋다",
        "어떤 단원을 마치고 나면 전체 내용을 다시 정리해본다", "외운것 같아도 막상 기억하려고 하면 기억나질 알아서 책을 뒤적인다",
        "문제풀이 할때는 물어보는 것이 무엇인지 먼저 파악하고 풀이를 시작한다"
    ],
    "학습 방법 및 기술 (5/7)": [
        "공부할 때는 별로 목표수립은 필요없다", "오늘 해야 할 공부가 계획되어 있다", "게임이나 친구들과 놀이 때문에 계획한 공부시간을 놓치는 경우가 많다",
        "같은 실수 때문에 잘못을 반복한다고 혼나는 경우가 많다", "선생님 수업 내용을 어렵지 않게 이해할 수 있다",
        "공부할 때 배우는 내용이 내가 알던 것과 달라서 의문을 가져본 적이 없다", "전체 내용을 보지 않고 밑줄 그은 것만 확인하며 공부해도 충분하다",
        "공식 같은것 외우지 않아도 충분히 좋은 성적을 받을 수 있다고 생각한다", "나올 만한 예상문제의 답만 외운 뒤 질문을 보자마자 답을 쓰는 경우가 많다"
    ],
    "학습 방법 및 기술 (6/7)": [
        "흥미 있는 대학 학과나 직업에 대해 이것 저것 찾아본 적이 있다", "단원평가나 학교시험의 범위와 일정에 맞춰 계획을 세워 공부한다",
        "공부하기 어려운 과목도 해야 할 분량은 빼먹지 않고 공부한다", "운이 나빠서 자꾸 일이 잘못 되는 것 같다",
        "새로 배우는 단원은 여러 번 반복해서 설명을 들어야 겨우 무슨 내용인지 알 수 있다", "새로운 것을 배우면 이전에 배운 내용들이 더욱 잘 이해되는 것 같다",
        "내 노트는 참고서를 복사한 것처럼 잘 정리되어 있다", "한 번 외운 내용은 오랫동안 잘 기억하는 편이다",
        "문제 풀 때에 무엇을 어떻게 활용해서 풀지 몰라 답답한 경우가 많다"
    ],
    "학습 방법 및 기술 (7/7)": [
        "목표가 있으면 부담스러워서 목표를 세우지 않고 공부하는 편이다", "계획은 어차피 바뀌므로 굳이 세울 필요가 없다",
        "책상 앞에 앉아 있지만 집중해서 공부한 시간은 얼마 되지 않는다", "잘된 일들은 굳이 되돌아볼 필요가 없다",
        "수업을 잘 듣지 못해도 참고서나 자습서를 이용하면 공부에 문제 없다", "참고서에 나온 내용도 왜 그런지 생각을 하면서 공부하는 편이다",
        "과목별로 일정한 나만의 노트 필기 방법으로 정리한다", "암기에 사용하는 노트가 따로 있다", "한 번 풀어서 맞춘 문제를 다음에 다시 풀어도 자주 틀리는 편이다"
    ]
}

# Part III: 학습동기
questions_part3 = {
    "공부하는 이유는? (1/3)": [
        "좋은 성적을 얻으면 용돈을 주시거나 좋은 선물을 사주니까", "남들 다 하는 공부 나만 안하면 불안해서", "공부하며 성장하는 내 모습이 자랑스러워서",
        "공부 안하면 어른들에게 잔소리 들으니까", "부모님이나 선생님이 기대하는 것을 만족시켜드리기 위해",
        "내가 원하는 직업을 얻기 위해서는 꼭 공부를 해야 하니까", "공부하는 것은 정말 싫지만 선생님이나 부모님이 하라고 하니까"
    ],
    "공부하는 이유는? (2/3)": [
        "학생은 당연히 공부를 해야 하니까", "공부하며 내가 몰랐던 것들을 알게 되는 것이 즐거워서", "공부하고 남은 시간을 자유롭게 보내기 위해",
        "부모님이나 선생님이 바라시는 대학에 가기위해", "내가 정한 목표를 하나씩 성취하는 것이 뿌듯해서",
        "좋은 성적을 받지 못하면 용돈이 줄거나 자유시간이 줄어서", "공부잘하면 다른 아이들이 나를 함부로 대하지 못하니까"
    ],
    "공부하는 이유는? (3/3)": [
        "시험 성적이 떨어지면 부모님께 혼나는 것이 싫어서",
        "공부잘해서 좋은 성적을 얻으면 다른 사람들이 칭찬해주니까", "다른 사람이 시켜서 하는 것보다 스스로 하는게 더 보람있으니까",
        "선생님이나 부모님이 공부하라고 한 분량을 맞춰놓아야 하니까", "가족들에게 모범이 되는 모습을 보여주어야 하니까",
        "공부하는 것은 그 누구보다 나에게 가장 도움이 되니까"
    ]
}

# --- 진행상황 저장/복원 함수들 ---
def save_progress(session_id: str, student_name: str, school_level: str, responses: dict):
    """검사 진행상황을 데이터베이스에 저장"""
    try:
        print(f"[DEBUG] 저장 시도 - 세션: {session_id}, 이름: {student_name}")
        db = SessionLocal()
        try:
            # 기존 진행상황 찾기
            progress = db.query(SurveyProgress).filter(SurveyProgress.session_id == session_id).first()
            
            # 완료된 문항 수 계산
            completed_count = sum(1 for v in responses.values() if v is not None and v != "")
            print(f"[DEBUG] 완료된 문항 수: {completed_count}")
            
            if progress:
                # 기존 기록 업데이트
                print(f"[DEBUG] 기존 기록 업데이트")
                progress.student_name = student_name
                progress.school_level = school_level
                progress.progress_data = json.dumps(responses, ensure_ascii=False)
                progress.completed = completed_count
                progress.last_updated = datetime.now()
            else:
                # 새 기록 생성
                print(f"[DEBUG] 새 기록 생성")
                progress = SurveyProgress(
                    session_id=session_id,
                    student_name=student_name,
                    school_level=school_level,
                    progress_data=json.dumps(responses, ensure_ascii=False),
                    completed=completed_count,
                    total_questions=150
                )
                db.add(progress)
            
            db.commit()
            print(f"[DEBUG] 저장 성공")
            return True
        finally:
            db.close()
    except Exception as e:
        print(f"진행상황 저장 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_progress(session_id: str):
    """세션 ID로 저장된 진행상황 불러오기"""
    try:
        print(f"[DEBUG] 세션 ID 검색 시도: {session_id}")
        db = SessionLocal()
        try:
            progress = db.query(SurveyProgress).filter(SurveyProgress.session_id == session_id).first()
            if progress:
                print(f"[DEBUG] 세션 찾음: {progress.student_name}, 완료: {progress.completed}")
                return {
                    'student_name': progress.student_name or "",
                    'school_level': progress.school_level or "초등",
                    'responses': json.loads(progress.progress_data),
                    'completed': progress.completed,
                    'total_questions': progress.total_questions,
                    'last_updated': progress.last_updated.strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                print(f"[DEBUG] 세션을 찾을 수 없음: {session_id}")
            return None
        finally:
            db.close()
    except Exception as e:
        print(f"진행상황 불러오기 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_session_id():
    """세션 ID 생성"""
    return str(uuid.uuid4())

def create_final_survey():
    with gr.Blocks(title="종합 학습 진단 검사", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 종합 학습 진단 검사")
        
        # 세션 관리 (숨겨진 상태)
        session_id = gr.State(value=generate_session_id())
        
        # 진행상황 및 옵션
        with gr.Row():
            with gr.Column(scale=2):
                sample_checkbox = gr.Checkbox(label="🎯 샘플 데이터로 테스트하기", value=False, info="체크하면 모든 설문이 임의의 값으로 자동 채워집니다")
            with gr.Column(scale=1):
                progress_info = gr.Markdown("📊 **진행률**: 0/150 (0%)")
            
        # 현재 세션 ID 표시 (세션 입력 필드 바로 위)
        current_session_display = gr.Markdown("")
            
        with gr.Row():
            session_input = gr.Textbox(label="세션 ID", placeholder="이전 검사를 이어하려면 세션 ID를 입력하세요", max_lines=1, scale=3)
            load_progress_btn = gr.Button("💾 이전 진행상황 불러오기", scale=1)
            with gr.Column(scale=1):
                save_status = gr.Markdown("")

        # 이름 입력 필드 및 학교급 선택
        with gr.Row():
            name_input = gr.Textbox(label="이름", placeholder="검사자 이름을 입력해주세요", max_lines=1, scale=2)
            school_level = gr.Dropdown(
                label="학교급", 
                choices=["초등", "중등", "고등"], 
                value="초등",
                scale=1
            )

        with gr.Row():
            # 좌측: 설문 영역
            with gr.Column(scale=3):
                gr.Markdown("### 📝 학습 성향 검사")

                all_responses: Dict[str, gr.Radio] = {}
                question_texts: List[str] = []
                options = ["아니다", "조금 아니다", "조금 그렇다", "그렇다"]

                # 질문 UI 동적 생성
                question_sets = [
                    ("Part I: 학업관련 감정과 행동 패턴", questions_part1, "자신의 생각이나 행동과 가장 가깝다고 느끼는 곳에 표시해주세요."),
                    ("Part II: 학습 방법 및 기술", questions_part2, "자신의 공부 습관과 가장 가깝다고 느끼는 곳에 표시해주세요."),
                    ("Part III: 학습동기", questions_part3, "내가 왜 공부하는지, 그 이유와 가장 가깝다고 느끼는 곳에 표시해주세요.")
                ]

                for title, questions, instruction in question_sets:
                    gr.Markdown(f"## {title}")
                    gr.Markdown(instruction)
                    for section, qs in questions.items():
                        gr.Markdown(f"### {section}")
                        for i, q_text in enumerate(qs):
                            key = f"{title}_{section}_{i}"
                            all_responses[key] = gr.Radio(options, label=q_text)
                            question_texts.append(q_text)

                submit_btn = gr.Button("제출", variant="primary")
                output_text = gr.Textbox(label="처리 상태", interactive=False, placeholder="모든 문항에 답변 후 제출 버튼을 눌러주세요.")
                report_output = gr.Markdown(label="학습 성향 분석 보고서", visible=False)

            # 우측: 채팅 영역
            with gr.Column(scale=2):
                gr.Markdown("### 🤖 AI 학습 도우미")
                gr.Markdown("검사 결과를 바탕으로 학습에 대해 궁금한 것을 물어보세요!")

                chatbot = gr.Chatbot(label="대화", height=500, show_label=True)
                image_input = gr.Image(label="이미지 업로드 (수학 문제, 과제 등)", type="filepath", height=150)

                with gr.Row():
                    chat_input = gr.Textbox(label="메시지 입력", placeholder="학습에 대해 궁금한 것을 물어보세요...", lines=1, scale=4)
                    chat_send = gr.Button("전송", variant="secondary", scale=1)

                gr.Markdown("💡 *팁: 검사를 완료하면 더 정확한 맞춤 조언을 받을 수 있어요!*")
                gr.Markdown("📷 *이미지로 수학 문제나 과제를 업로드하면 단계별로 도움을 받을 수 있어요!*")

        # 샘플 데이터 자동 채우기 함수
        def fill_sample_data(use_sample):
            if use_sample:
                # 각 설문에 임의의 값(1-4) 할당
                options = ["아니다", "조금 아니다", "조금 그렇다", "그렇다"]
                updates = []
                for _ in all_responses:
                    random_choice = random.choice(options)
                    updates.append(gr.update(value=random_choice))
                return updates
            else:
                # 체크 해제 시 모든 값을 None으로 초기화
                updates = []
                for _ in all_responses:
                    updates.append(gr.update(value=None))
                return updates

        def update_progress_info(*responses):
            """진행률 정보 업데이트"""
            completed = sum(1 for r in responses if r is not None and r != "")
            total = len(responses)
            percentage = round((completed / total) * 100) if total > 0 else 0
            return f"📊 **진행률**: {completed}/{total} ({percentage}%)"
        
        def auto_save_progress(session_id, name, school_level_value, sample_checkbox, *responses):
            """자동 저장 (응답 변경 시마다 호출)"""
            # 샘플 데이터 모드이면 저장하지 않음
            if sample_checkbox:
                return ""
                
            if name and name.strip():  # 이름이 입력된 경우에만 저장
                response_dict = {}
                for i, response in enumerate(responses):
                    if i < len(question_texts):
                        response_dict[question_texts[i]] = response
                
                if save_progress(session_id, name.strip(), school_level_value, response_dict):
                    completed = sum(1 for r in responses if r is not None and r != "")
                    return f"💾 자동 저장됨 ({completed}/150)"
                else:
                    return "❌ 저장 실패"
            return ""
        
        def show_current_session_id(session_id_value):
            """현재 세션 ID 표시"""
            return f"🔑 **현재 세션**: `{session_id_value}`\n💡 위 ID를 저장해두시면 나중에 이어서 검사할 수 있습니다! 이름부터 쓰세요."
        
        def load_previous_progress(session_input_value):
            """이전 진행상황 불러오기"""
            if not session_input_value or not session_input_value.strip():
                return [gr.update() for _ in all_responses] + [gr.update(), gr.update(), "세션 ID를 입력해주세요"]
            
            progress_data = load_progress(session_input_value.strip())
            if progress_data:
                # 응답 데이터 복원
                updates = []
                loaded_responses = progress_data['responses']
                for question in question_texts:
                    value = loaded_responses.get(question, None)
                    updates.append(gr.update(value=value))
                
                # 이름과 학교급 복원
                name_update = gr.update(value=progress_data['student_name'])
                school_update = gr.update(value=progress_data['school_level'])
                
                # 상태 메시지
                status_msg = f"✅ 진행상황 복원 완료! (마지막 저장: {progress_data['last_updated']})"
                
                return updates + [name_update, school_update, status_msg]
            else:
                return [gr.update() for _ in all_responses] + [gr.update(), gr.update(), "❌ 해당 세션을 찾을 수 없습니다. 먼저 이름을 입력하고 설문에 답변하여 진행상황을 저장해주세요."]

        def submit(session_id_value, name, school_level_value, *responses):
            if not name or not name.strip():
                return "오류: 이름을 입력해주세요.", gr.update(visible=False)

            if None in responses:
                none_index = responses.index(None)
                unanswered_question = question_texts[none_index]
                return f"'{unanswered_question}' 질문에 답변해주세요.", gr.update(visible=False)

            try:
                # Gradio 응답(문자열)을 점수(숫자)로 변환
                to_score = {"아니다": 1, "조금 아니다": 2, "조금 그렇다": 3, "그렇다": 4}
                scored_responses = {q_text: to_score[resp] for q_text, resp in zip(question_texts, responses)}

                # 1. 원점수 계산 (esli_01)
                # calculate_scores가 scored_responses 딕셔너리를 처리할 수 있도록 수정되었다고 가정
                raw_scores_df = calculate_scores(scored_responses)
                
                # 2. 보고서 생성 및 DB 저장 (esli_02)
                # generate_report_with_llm이 scored_responses 딕셔너리와 학교급을 함께 받는다고 가정
                report_content = generate_report_with_llm(student_name=name.strip(), responses=scored_responses, school_level=school_level_value)

                if "데이터베이스 저장에 실패했습니다" in report_content or "[LLM 코멘트 생성 실패" in report_content:
                     return f"보고서 생성 중 일부 오류가 발생했습니다. 하지만 생성된 내용은 다음과 같습니다.", gr.update(value=report_content, visible=True)
                
                return f"✅ 분석이 완료되었습니다! 아래에서 결과를 확인하세요.\n\n📋 **이 세션의 ID**: `{session_id_value}` (향후 이어서 하기용)", gr.update(value=report_content, visible=True)

            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"분석 처리 중 심각한 오류 발생: {e}", gr.update(visible=False)

        def chat_respond(message, history, image, name):
            if not (message and message.strip()) and not image:
                return history, "", None # 메시지와 이미지가 모두 없으면 아무것도 하지 않음
            
            # 이름이 없으면 채팅 불가 안내
            student_name = name.strip() if name and name.strip() else None
            if not student_name:
                history.append((message, "원활한 상담을 위해 먼저 설문조사를 완료하고 이름을 입력해주세요."))
                return history, "", None

            # esli_03의 채팅 함수 호출
            response = gradio_chat_with_history(message, history, image, student_name)
            history.append((message, response))
            return history, "", None # 입력창과 이미지 업로드 초기화

        # 이벤트 바인딩
        all_components = [session_id, name_input, school_level] + list(all_responses.values())
        submit_btn.click(fn=submit, inputs=all_components, outputs=[output_text, report_output])

        # 샘플 체크박스 이벤트 바인딩
        sample_checkbox.change(
            fn=fill_sample_data,
            inputs=[sample_checkbox],
            outputs=list(all_responses.values())
        )
        
        # 진행상황 자동 저장 및 진행률 업데이트 (응답 변경 시마다)
        for response_component in all_responses.values():
            response_component.change(
                fn=update_progress_info,
                inputs=list(all_responses.values()),
                outputs=[progress_info]
            )
            # 이름이 입력된 경우 자동 저장
            response_component.change(
                fn=auto_save_progress,
                inputs=[session_id, name_input, school_level, sample_checkbox] + list(all_responses.values()),
                outputs=[save_status]
            )
        
        # 이름이나 학교급 변경 시에도 자동 저장
        name_input.change(
            fn=auto_save_progress,
            inputs=[session_id, name_input, school_level, sample_checkbox] + list(all_responses.values()),
            outputs=[save_status]
        )
        school_level.change(
            fn=auto_save_progress,
            inputs=[session_id, name_input, school_level, sample_checkbox] + list(all_responses.values()),
            outputs=[save_status]
        )
        
        # 진행상황 불러오기 버튼
        load_progress_btn.click(
            fn=load_previous_progress,
            inputs=[session_input],
            outputs=list(all_responses.values()) + [name_input, school_level, save_status]
        )
        
        # 페이지 로드 시 현재 세션 ID 표시
        demo.load(
            fn=show_current_session_id,
            inputs=[session_id],
            outputs=[current_session_display]
        )

        chat_send.click(
            fn=chat_respond,
            inputs=[chat_input, chatbot, image_input, name_input],
            outputs=[chatbot, chat_input, image_input]
        )
        chat_input.submit(
            fn=chat_respond,
            inputs=[chat_input, chatbot, image_input, name_input],
            outputs=[chatbot, chat_input, image_input]
        )

    return demo

if __name__ == "__main__":
    # 데이터베이스 초기화 (새 테이블 포함)
    init_db()
    
    survey_app = create_final_survey()
    # Gradio 처리큐로 LLM 작업 폭주 방지 (동시 처리 10)
    survey_app = survey_app.queue(concurrency_count=10)
    port = int(os.getenv("PORT", 7861))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Render 환경에서는 FastAPI로 마운트
    is_render = os.getenv("RENDER", "false").lower() == "true"
    
    if is_render:
        # Render 환경: FastAPI로 감싸서 실행
        from fastapi import FastAPI, Response
        from fastapi.responses import RedirectResponse
        import uvicorn
        
        app = FastAPI()
        
        # 헬스체크 엔드포인트 추가
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "learning-assessment"}
        
        @app.get("/")
        async def root():
            # 루트 경로 접속 시 Gradio 앱으로 리다이렉트
            return RedirectResponse(url="/app", status_code=302)

        # Render의 내부 헬스체크가 HEAD / 를 칠 때 405가 되지 않도록 보완
        @app.head("/")
        async def root_head():
            return Response(status_code=200)
        
        app = gr.mount_gradio_app(app, survey_app, path="/app")
        
        # uvloop/httptools 사용 시 약간의 성능 향상
        try:
            import uvloop  # noqa: F401
            import httptools  # noqa: F401
            uvicorn.run(app, host="0.0.0.0", port=port, loop="uvloop", http="httptools")
        except Exception:
            uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # 로컬 환경: 기존 방식
        survey_app.launch(
            server_name=host,
            server_port=port,
            share=False,
            inbrowser=False
        )