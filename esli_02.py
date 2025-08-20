import pandas as pd
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
import json

# 데이터베이스 연동을 위한 import
from database import SessionLocal, SurveyResponse, ReferenceStandard, ReferencePercentile

load_dotenv()
# 환경 변수에서 OpenAI API 키를 읽어옵니다
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_llm_for_report(prompt):
    """
    OpenAI의 LLM을 호출하여 프롬프트에 대한 맞춤형 보고서 내용을 생성합니다.
    """
    # API 키가 설정되지 않은 경우 예외 처리
    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
        print("--- [경고] OpenAI API 키가 설정되지 않았습니다. 예시 코멘트를 생성합니다. ---")
        # 각 프롬프트 유형에 맞는 예시 응답 반환
        if "학습 동기" in prompt:
            return """#### 검사 결과 분석
학생의 학습 동기는 '현상 유지적 학습형'의 특징을 보입니다. 스스로 성장하고 싶은 마음(자기 성취 T=108)이 주변의 기대를 충족시키려는 마음(사회적 관계 T=101)보다 앞서고 있으며, 이는 긍정적인 신호입니다. 다만, 아직은 '무엇을 위해' 공부해야 하는지에 대한 구체적인 그림이 그려지지 않아, 지금의 상태에 머무르려는 경향이 있을 수 있습니다.

#### 코칭 코멘트
**"더 나은 내일을 꿈꾸는 마음, 구체적인 목표로 날개를 달아주세요."**

* **현재 모습**: 학생은 더 나은 미래를 막연히 그리며 공부의 필요성을 인지하고 있습니다. 하지만 뚜렷한 목표가 없어 열정적으로 학습에 몰입하기보다는, 익숙한 수준의 노력을 유지하는 모습을 보일 수 있습니다.
* **성장의 기회**: 이 시기는 자신이 진정으로 무엇을 좋아하고 원하는지 탐색하는 매우 중요한 과정입니다. 조급해할 필요 없이, 이 탐색 과정을 통해 자신만의 학습 이유를 찾는다면 폭발적인 성장을 이룰 수 있습니다.
* **코칭 제안**: 학생이 흥미를 느끼는 분야에 대해 함께 대화하며 관련 정보(직업, 학과, 관련 인물 등)를 찾아보는 활동을 추천합니다. 이를 통해 막연했던 미래가 구체적인 목표로 바뀌고, '왜 공부해야 하는지'에 대한 강력한 내적 동기로 발전할 수 있도록 지지하고 격려해주시기 바랍니다."""
        elif "학습 전략" in prompt:
            return """#### 검사 결과 분석
학생은 학습 내용을 깊이 있게 파고드는 '사고력'(T=107)이라는 강력한 무기를 가지고 있지만, 이를 성과로 연결할 '학습 전략'(T=86)이 다소 부족한 상황입니다. 특히, 학습의 방향키 역할을 하는 '목표 세우기'(T=66)에서 어려움을 겪고 있어, 뛰어난 잠재력에도 불구하고 어디로 나아가야 할지 막막함을 느낄 수 있습니다.

#### 코칭 코멘트
**"내 안의 좋은 무기들, '전략'이라는 지도를 들고 사용해 보세요."**

* **현재 모습**: 학생은 체계적인 계획 없이 그때그때 주어진 공부를 하는 경향이 있습니다. 생각하는 힘은 좋지만, 학습의 전체적인 그림을 그리지 못해 노력에 비해 아쉬운 결과를 얻을 수 있습니다.
* **성장의 기회**: 이미 보유한 '사고력'과 평균 수준의 '학습 기술'은 훌륭한 자산입니다. 여기에 '목표 설정'과 '계획 수립'이라는 전략적 측면만 보완된다면, 학습 효율과 성과가 눈에 띄게 향상될 것입니다.
* **코칭 제안**:
    1.  **나만의 나침반 만들기**: '버킷리스트'나 '만다라트 계획표'를 활용해 장기적, 단기적 목표를 시각화하는 활동을 추천합니다. 거창하지 않아도 좋습니다. '한 달 안에 수학 문제집 20쪽 풀기'처럼 작고 구체적인 목표부터 시작해 성취감을 느끼는 것이 중요합니다.
    2.  **가장 약한 고리 강화하기**: 학습 기술 중 상대적으로 아쉬운 '이해하기'와 '문제 풀기' 능력을 보완해야 합니다. 공부 시작 전, 단원의 목표를 먼저 읽어보는 습관을 들이고, 다양한 유형의 문제를 접하며 배운 개념을 적용하는 연습을 꾸준히 해나가는 것이 좋습니다."""
        else: # 학습 방해 요인
            return """#### 검사 결과 분석
학생은 학습을 방해하는 특별한 심리적, 행동적 어려움 없이 매우 안정적인 상태를 유지하고 있습니다. 스트레스에 대한 대처 능력, 자신감, 주변 환경과의 관계 등 모든 면에서 긍정적인 모습을 보여주고 있어, 학습에 집중할 수 있는 훌륭한 기반을 갖추고 있습니다.

#### 코칭 코멘트
**"이미 튼튼하게 다져진 마음의 땅, 이제 성장의 씨앗을 심을 때입니다."**

* **현재 모습**: 학생은 정서적으로 안정되어 있고, 자기 통제력 또한 양호하여 학습 외적인 요인으로 인해 흔들리는 일이 적습니다. 이는 공부에 온전히 집중할 수 있는 큰 강점입니다.
* **성장의 기회**: 현재 학생에게 가장 필요한 것은 '왜 공부하는가(동기)'에 대한 답을 찾고, '어떻게 공부할 것인가(전략)'에 대한 구체적인 계획을 세우는 것입니다.
* **코칭 제안**: 지금의 안정된 심리 상태와 좋은 행동 습관이라는 강점을 적극 활용해야 합니다. 이 튼튼한 기반 위에서, 앞서 제안된 '학습 동기'와 '학습 전략'을 보완해 나간다면, 학생의 잠재력이 기대 이상으로 발휘될 수 있을 것입니다. 자신의 강점을 믿고 새로운 도전을 시작할 수 있도록 격려해주시기 바랍니다."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # 또는 "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "당신은 학생의 학습 성향 데이터를 분석하고 조언하는 전문 학습 코치입니다. 주어진 데이터를 기반으로, 학생에게 친절하고 지지적이지만, 전문적인 말투를 사용해 독창적인 보고서를 작성해 주세요. 딱딱한 설명서가 아닌, 학생의 성장을 돕는 따뜻한 조언의 느낌을 담아주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"--- OpenAI API 호출 중 오류 발생: {e} ---")
        return f"--- [LLM 코멘트 생성 실패: {e}] ---"


def generate_report_with_llm(student_name: str, responses: dict):
    """
    학생 데이터를 분석하고 LLM을 호출하여 맞춤형 보고서를 생성하고, 결과를 DB에 저장합니다.
    """
    db = SessionLocal()
    try:
        # --- 1. 데이터 로드 및 계산 ---
        # CSV 파일 로드 대신, Gradio 앱에서 직접 받은 responses 딕셔너리를 사용합니다.
        # 참조값은 DB에서 조회 (기본: 초등 기준)
        ref_level = os.getenv("REF_LEVEL", "초등")
        session = SessionLocal()
        try:
            std_rows = session.query(ReferenceStandard).filter(ReferenceStandard.level == ref_level).all()
            if not std_rows:
                return f"오류: 기준표(표준점수-{ref_level})가 DB에 없습니다. database.py의 seed_reference_data를 실행하세요."
            std_info_df = pd.DataFrame(
                {"평균": {r.name: r.mean for r in std_rows}, "표준편차": {r.name: r.std for r in std_rows}}
            )
            pct_rows = session.query(ReferencePercentile).all()
            percentile_df = pd.DataFrame(
                {"표준점수": [r.t_score for r in pct_rows], "백분위": [r.percentile for r in pct_rows]}
            ).set_index("표준점수")
        finally:
            session.close()

        COLUMN_MAP = {
            '직접적': '직접적 보상처벌', '관계적': '사회적 관계', '자기성취': '자기성취',
            '스트레스': '스트레스민감성', '효능감': '학습효능감', '친구': '친구관계',
            '가정': '가정환경', '학교': '학교환경', '수면': '수면조절',
            '집중력': '학습집중력', 'TV': 'TV프로그램', '컴퓨터': '컴퓨터',
            '스마트': '스마트기기', '학습전략': '학습전략', '학습기술': '학습기술',
            '목표': '목표세우기', '계획': '계획하기', '실천': '실천하기',
            '돌아보기': '돌아보기', '이해하기': '이해하기', '사고하기': '사고하기',
            '정리하기': '정리하기', '암기하기': '암기하기', '문제풀기': '문제풀기',
        }

        # 전달받은 responses로부터 항목별 원점수 집계
        # responses: {질문텍스트: 1~4}
        # 간단히 항목명의 키워드를 포함하는 질문들을 평균하여 원점수 근사
        # 실제 설문-항목 매핑은 esli_01.py 개선 후 교체
        student_raw_scores = {}
        keyword_to_name = {
            "목표": "목표세우기", "계획": "계획하기", "실천": "실천하기", "돌아보": "돌아보기",
            "이해": "이해하기", "사고": "사고하기", "정리": "정리하기", "암기": "암기하기", "문제": "문제풀기",
            "스트레스": "스트레스민감성", "효능": "학습효능감", "친구": "친구관계", "가정": "가정환경",
            "학교": "학교환경", "수면": "수면조절", "집중": "학습집중력", "TV": "TV프로그램",
            "컴퓨터": "컴퓨터", "스마트": "스마트기기",
            # 동기 요인 근사
            "보상": "직접적 보상처벌", "관계": "사회적 관계", "성취": "자기성취"
        }
        buckets = {}
        for q, val in responses.items():
            for kw, name in keyword_to_name.items():
                if kw in q:
                    buckets.setdefault(name, []).append(val)
        for name, vals in buckets.items():
            if len(vals) > 0:
                student_raw_scores[name] = float(sum(vals)) / len(vals) * 25  # 1~4척도 → 100점 환산 근사


        student_scores = {}
        for col_alias, col_std_name in COLUMN_MAP.items():
            if col_alias in student_raw_scores and col_std_name in std_info_df.index:
                raw_score = student_raw_scores[col_alias]
                mean = std_info_df.loc[col_std_name, '평균']
                std = std_info_df.loc[col_std_name, '표준편차']
                t_score = round(100 + 15 * ((raw_score - mean) / std))
                percentile = percentile_df.loc[t_score, '백분위'] if t_score in percentile_df.index else "N/A"
                student_scores[col_std_name] = {'raw': raw_score, 't_score': t_score, 'percentile': int(percentile) if percentile != "N/A" else 0}

        # --- 2. 보고서 각 섹션별 LLM 프롬프트 생성 및 호출 (기존 코드와 동일) ---
        m_type, _, m_reason, m_coaching = get_motivation_analysis(student_scores)
        motivation_prompt = f"""
        '학습 동기'에 대한 분석 및 코칭 코멘트를 작성해줘.

        [학생 데이터 요약]
        - 학생 이름: {student_name}
        - 주요 동기 유형: {m_type}
        - 자기 성취 동기 점수: T점수 {student_scores['자기성취']['t_score']} (백분위 {student_scores['자기성취']['percentile']}%)
        - 사회적 관계 동기 점수: T점수 {student_scores['사회적 관계']['t_score']} (백분위 {student_scores['사회적 관계']['percentile']}%)
        - 직접적 보상/처벌 동기 점수: T점수 {student_scores['직접적 보상처벌']['t_score']} (백분위 {student_scores['직접적 보상처벌']['percentile']}%)

        [참고 가이드라인]
        - 핵심 특징: {m_reason}
        - 코칭 방향: {m_coaching}

        [작성 지침]
        - 위의 데이터와 가이드라인을 '참고'하여, 너만의 독창적이고 전문적인 코멘트를 생성해줘.
        - 딱딱한 설명이 아닌, 학생의 마음을 이해하고 성장을 지지하는 따뜻한 조언의 형태로 작성해줘.
        - 아래 출력 형식을 반드시 지켜줘.

        [출력 형식]
        #### 검사 결과 분석
        (여기에 데이터 기반의 객관적인 분석 작성)

        #### 코칭 코멘트
        "**여기에 한 줄 요약 코멘트 작성**"
        * **현재 모습**: (여기에 학생의 현재 상태 묘사)
        * **성장의 기회**: (여기에 긍정적 측면과 성장 가능성 묘사)
        * **코칭 제안**: (여기에 구체적인 조언 작성)
        """
        motivation_comment = call_llm_for_report(motivation_prompt)

        # 2-2. 학습 전략/기술 프롬프트
        s_analysis, s_coaching_title = get_strategy_analysis(student_scores)
        strategy_prompt = f"""
        '학습 전략/기술'에 대한 분석 및 코칭 코멘트를 작성해줘.

        [학생 데이터 요약]
        - 종합 분석: {s_analysis}
        - 학습 전략 종합 점수: T점수 {student_scores['학습전략']['t_score']} (백분위 {student_scores['학습전략']['percentile']}%)
        - 학습 기술 종합 점수: T점수 {student_scores['학습기술']['t_score']} (백분위 {student_scores['학습기술']['percentile']}%)
        - 강점 항목: 사고하기 (T={student_scores['사고하기']['t_score']})
        - 약점 항목: 목표세우기 (T={student_scores['목표세우기']['t_score']}), 이해하기 (T={student_scores['이해하기']['t_score']}), 문제풀기 (T={student_scores['문제풀기']['t_score']})

        [참고 가이드라인]
        - 핵심 특징: {s_analysis}
        - 코칭 방향: "{s_coaching_title}" 이 제목에 어울리는 내용으로, 전략(목표/계획) 보완과 기술(이해/문제풀이) 강화를 조언해줘.

        [작성 지침]
        - 강점(사고력)을 인정해주고, 약점(전략)을 보완하면 더 크게 성장할 수 있다는 점을 강조해줘.
        - '버킷리스트', '만다라트 계획표' 등 구체적인 활동 예시를 들어 조언해줘.
        - 아래 출력 형식을 반드시 지켜줘.

        [출력 형식]
        #### 검사 결과 분석
        (여기에 데이터 기반의 객관적인 분석 작성)

        #### 코칭 코멘트
        "**여기에 한 줄 요약 코멘트 작성**"
        * **현재 모습**: (여기에 학생의 현재 상태 묘사)
        * **성장의 기회**: (여기에 긍정적 측면과 성장 가능성 묘사)
        * **코칭 제안**: (여기에 구체적인 조언을 1, 2번으로 나누어 작성)
        """
        strategy_comment = call_llm_for_report(strategy_prompt)

        # 2-3. 학습 방해 요인 프롬프트
        h_analysis, h_coaching_title = get_hindrance_analysis(student_scores)
        hindrance_prompt = f"""
        '학습 방해 심리/행동'에 대한 분석 및 코칭 코멘트를 작성해줘.

        [학생 데이터 요약]
        - 종합 분석: {h_analysis}
        - 주요 점수: 스트레스민감성(T={student_scores['스트레스민감성']['t_score']}), 학습효능감(T={student_scores['학습효능감']['t_score']}), 학습집중력(T={student_scores['학습집중력']['t_score']})
        - 모든 방해 요인 점수가 안정적인 범위에 있음.

        [참고 가이드라인]
        - 핵심 특징: {h_analysis}
        - 코칭 방향: "{h_coaching_title}" 이 제목처럼, 이미 훌륭한 기반을 갖추고 있음을 칭찬하고, 이 강점을 활용하여 동기와 전략을 키워나가도록 격려해줘.

        [작성 지침]
        - 학생이 이미 가진 강점(정서적 안정, 자기 통제력)을 구체적으로 칭찬하며 자신감을 심어줘.
        - 이 튼튼한 기반 위에서 다른 영역(동기, 전략)을 발전시켜 나갈 때임을 강조해줘.
        - 아래 출력 형식을 반드시 지켜줘.

        [출력 형식]
        #### 검사 결과 분석
        (여기에 데이터 기반의 객관적인 분석 작성)

        #### 코칭 코멘트
        "**여기에 한 줄 요약 코멘트 작성**"
        * **현재 모습**: (여기에 학생의 현재 상태 묘사)
        * **성장의 기회**: (여기에 긍정적 측면과 성장 가능성 묘사)
        * **코칭 제안**: (여기에 구체적인 조언 작성)
        """
        hindrance_comment = call_llm_for_report(hindrance_prompt)


        # --- 3. 최종 보고서 조합 ---
        score_table_md = "| 구분 | 영역 | 원점수 | 표준점수(T) | 백분위(%) |\n"
        score_table_md += "| :--- | :--- | :--- | :--- | :--- |\n"
        categories_in_order = {
            "학습 동기": ['직접적 보상처벌', '사회적 관계', '자기성취'],
            "학습 전략": ['목표세우기', '계획하기', '실천하기', '돌아보기', '학습전략'],
            "학습 기술": ['이해하기', '사고하기', '정리하기', '암기하기', '문제풀기', '학습기술'],
            "학습 방해 (심리)": ['스트레스민감성', '학습효능감', '친구관계', '가정환경', '학교환경'],
            "학습 방해 (행동)": ['수면조절', '학습집중력', 'TV프로그램', '컴퓨터', '스마트기기']
        }
        for group, items in categories_in_order.items():
            for item in items:
                if item in student_scores:
                    score_data = student_scores[item]
                    item_name = '종합' if item in ['학습전략', '학습기술'] else item.replace('세우기','').replace('하기','')
                    score_table_md += f"| **{group.split(' ')[0]}** | {item_name} | {score_data['raw']} | {score_data['t_score']} | {score_data['percentile']} |\n"

        # 최종 보고서 텍스트
        report_md = f"""# {student_name} 학생 학습 성향 분석 종합 보고서 (LLM 기반)

        ---

        ## Ⅰ. 검사 결과 요약

        학생의 원점수를 전국 학생 데이터와 비교한 표준점수(T점수)와 백분위 점수입니다. 이를 통해 각 항목의 상대적인 강점과 약점을 객관적으로 파악할 수 있습니다.

        {score_table_md}
        ---

        ## Ⅱ. 학습 성향 종합 분석 및 코칭

        ### 1. 학습 동기: 무엇이 나의 공부를 이끌고 있는가?

        {motivation_comment}

        ---

        ### 2. 학습 전략/기술: 나는 어떻게 공부하고 있는가?

        {strategy_comment}

        ---

        ### 3. 학습 방해 요인: 내 공부를 막는 것은 없는가?

        {hindrance_comment}
        """

        # --- 4. 결과를 데이터베이스에 저장 ---
        try:
            new_response = SurveyResponse(
                student_name=student_name,
                responses_json=json.dumps(responses, ensure_ascii=False),
                scores_json=json.dumps(student_scores, ensure_ascii=False),
                report_content=report_md
            )
            db.add(new_response)
            db.commit()
            db.refresh(new_response)
            print(f"--- [성공] {student_name} 학생의 검사 결과가 데이터베이스에 저장되었습니다. (ID: {new_response.id}) ---")
            return report_md # 성공 시 생성된 보고서 내용을 반환
        except Exception as e:
            db.rollback()
            print(f"--- [오류] 데이터베이스 저장 중 오류 발생: {e} ---")
            # DB 저장에 실패하더라도 보고서 내용은 반환하여 사용자에게 보여줄 수 있도록 함
            return f"데이터베이스 저장에 실패했습니다. 하지만 보고서는 생성되었습니다.\n\n{report_md}"

    finally:
        db.close()


# --- 도우미 함수들 (규칙 기반 분석 로직) ---
def get_motivation_analysis(scores):
    t_self_achieve = scores['자기성취']['t_score']
    t_social = scores['사회적 관계']['t_score']
    t_reward_punish = scores['직접적 보상처벌']['t_score']
    if t_self_achieve > 114:
        return "자기 주도적 학습형", "자기 주도적 학습형에 가장 가까운 특성을 보이고 있습니다.", "자신이 원하는 이상이나 직업 등, 자신이 관심 있는 분야의 호기심을 충족시키기 위해 공부를 합니다.", "학습에 대한 호기심을 꾸준히 가질 수 있도록, 학생이 스스로 찾고 노력하는 것을 자주 격려하고 지지해 주시기 바랍니다."
    elif t_social > 114:
        return "사회 기대적 학습형", "사회 기대적 학습형에 가장 가까운 특성을 보이고 있습니다.", "다른 사람에게 좋은 모습을 보여주어야 한다는 생각에 공부를 합니다.", "자신이 타인의 기대에 부응하고 있다고 느끼도록 해주시기 바랍니다."
    elif t_self_achieve > 85:
        return "현상 유지적 학습형", "현상 유지적 학습형에 가장 가까운 특성을 보이고 있습니다.", "막연히 지금보다 나은 사람이 되거나 나의 미래를 준비하기 위해 공부하고자 생각합니다.", "적극적으로 학생이 원하는 것을 찾을 수 있도록 도와주는 것이 필요합니다."
    elif t_social > 85:
        return "군중 심리적 학습형", "군중 심리적 학습형에 가장 가까운 특성을 보이고 있습니다.", "다른 친구들이 모두 공부를 할 때 자신도 공부 하지 않으면 뒤쳐질 것이란 생각으로 공부를 하는 학생들입니다.", "자신이 어떤 사람이고 무엇을 좋아하는지 탐색하면서, 자기 자신을 믿을 수 있도록 끊임없이 칭찬과 격려를 해주는 것이 필요합니다."
    elif t_reward_punish > 114:
        return "타인 주도적 학습형", "타인 주도적 학습형에 가장 가까운 특성을 보이고 있습니다.", "용돈이나 선물처럼 누가 약속한 물질적인 보상을 얻기 위해 공부하거나, 공부를 하지 않아서 혼이 나는 것을 피하기 위한 목적으로 공부를 합니다.", "학생이 공부를 하는 것을 인정해주고 칭찬하여 꼭 눈에 보이는 보상이 아니더라도 공부를 통해 만족감을 얻을 수 있도록 해주는 것이 좋습니다."
    else:
        return "학습 동기 부재형", "학습 동기 부재형에 가장 가까운 특성을 보이고 있습니다.", "공부를 하는 이유가 그 어떤 것을 통해서도 생기지 않는 경우입니다.", "학생이 좋아하는 것이 무엇인지 찾아보고, 좋아하는 것과 공부가 연결될 수 있는 고리를 찾아보시기 바랍니다."

def get_strategy_analysis(scores):
    t_strategy = scores['학습전략']['t_score']
    t_skill = scores['학습기술']['t_score']
    def get_level(score):
        if score > 114: return '상'
        if score >= 86: return '중'
        return '하'
    strategy_level, skill_level = get_level(t_strategy), get_level(t_skill)
    analysis_map = {('상', '상'): "학습 전략과 기술이 모두 뛰어난 상태입니다.", ('상', '중'): "학습 전략 부분은 뛰어나지만, 학습 기술 부분은 일반적인 수준입니다.", ('상', '하'): "학습 전략 부분은 뛰어나지만, 학습 기술 부분이 취약합니다.", ('중', '상'): "학습 기술 부분은 뛰어나지만, 학습 전략 부분은 일반적인 수준입니다.", ('중', '중'): "학습 전략과 기술 부분 모두 일반적인 수준입니다.", ('중', '하'): "학습 전략 부분은 일반적인 반면, 학습 기술 부분이 취약합니다.", ('하', '상'): "학습 기술 부분은 뛰어나지만, 학습 전략 부분이 취약합니다.", ('하', '중'): "학습 기술 부분은 일반적인 반면, 학습 전략 부분이 취약합니다.", ('하', '하'): "학습 전략과 기술이 모두 취약한 상태입니다."}
    coaching_map = {('하', '중'): "생각하는 힘은 좋지만, 체계적인 학습 관리와 효율적인 공부법이 필요해요."}
    return analysis_map.get((strategy_level, skill_level), ""), coaching_map.get(('하', '중'), "맞춤형 코칭이 필요합니다.")

def get_hindrance_analysis(scores):
    psych_hindrance = any([scores['스트레스민감성']['t_score'] > 114, scores['학습효능감']['t_score'] < 86, scores['친구관계']['t_score'] < 86, scores['가정환경']['t_score'] < 86, scores['학교환경']['t_score'] < 86])
    behav_hindrance = any([scores['수면조절']['t_score'] < 86, scores['학습집중력']['t_score'] < 86, scores['TV프로그램']['t_score'] > 114, scores['컴퓨터']['t_score'] > 114, scores['스마트기기']['t_score'] > 114])
    if not psych_hindrance and not behav_hindrance: return "학습을 방해하는 심리적, 행동적 요인 모두 특별히 나쁜 영역 없이 긍정적인 상태를 보이고 있습니다.", "공부에 집중할 수 있는 좋은 마음과 행동 습관을 가지고 있어요."
    if psych_hindrance and not behav_hindrance: return "학습 방해 부분에서는 심리적 부분에서 좋지 않은 영향을 받고 있는 것 같습니다.", "심리적 안정감을 찾기 위한 노력이 필요합니다."
    if not psych_hindrance and behav_hindrance: return "학습 방해 부분에서는 행동적 부분에서 좋지 않은 영향을 받고 있는 것 같습니다.", "학습 습관을 개선하기 위한 노력이 필요합니다."
    return "학습 방해 부분에서는 심리적 부분과 행동적 부분 모두 좋지 않은 영향을 받고 있는 것 같습니다.", "심리적, 행동적 측면 모두 개선이 필요합니다."

# --- 메인 함수 실행 (테스트용으로 남겨두거나 삭제) ---
if __name__ == "__main__":
    # 테스트를 위해서는 'responses' 딕셔너리를 실제 데이터처럼 만들어야 합니다.
    # 예시: test_responses = {'문항1': 3, '문항2': 4, ...}
    # result_message = generate_report_with_llm("홍길동", test_responses)
    # print(result_message)
    print("esli_02.py 실행. 데이터베이스 연동 테스트를 위해서는 직접 함수를 호출해야 합니다.")
