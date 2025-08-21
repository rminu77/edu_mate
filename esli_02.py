import pandas as pd
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
import json
from typing import Optional, Dict, List, Tuple

# 데이터베이스 연동을 위한 import
from database import (
    SessionLocal,
    SurveyResponse,
    ReferenceStandard,
    ReferencePercentile,
    init_db,
    seed_reference_data,
    ReferenceQuestionMap,
    ReferenceQuestionUnmapped,
)

load_dotenv()
# 환경 변수에서 OpenAI API 키를 읽어옵니다
openai.api_key = os.getenv("OPENAI_API_KEY")


# --------------------
# 참조데이터 메모리 캐시
# --------------------
STD_INFO_CACHE: Dict[str, pd.DataFrame] = {}
PERCENTILE_DF_CACHE: Optional[pd.DataFrame] = None
QUESTION_MAP_CACHE: Optional[List[Tuple[str, str]]] = None


def get_std_info_df(level: str) -> pd.DataFrame:
    global STD_INFO_CACHE
    if level in STD_INFO_CACHE:
        return STD_INFO_CACHE[level]
    session = SessionLocal()
    try:
        rows = session.query(ReferenceStandard).filter(ReferenceStandard.level == level).all()
        if not rows:
            raise RuntimeError(f"오류: 기준표(표준점수-{level})가 DB에 없습니다. database.py의 seed_reference_data를 실행하세요.")
        df = pd.DataFrame({
            "평균": {r.name: r.mean for r in rows},
            "표준편차": {r.name: r.std for r in rows},
        })
        STD_INFO_CACHE[level] = df
        return df
    finally:
        session.close()


def get_percentile_df() -> pd.DataFrame:
    global PERCENTILE_DF_CACHE
    if PERCENTILE_DF_CACHE is not None:
        return PERCENTILE_DF_CACHE
    session = SessionLocal()
    try:
        pct_rows = session.query(ReferencePercentile).all()
        df = pd.DataFrame({
            "표준점수": [r.t_score for r in pct_rows],
            "백분위": [r.percentile for r in pct_rows],
        }).set_index("표준점수")
        PERCENTILE_DF_CACHE = df
        return df
    finally:
        session.close()


def get_question_map_pairs() -> List[Tuple[str, str]]:
    global QUESTION_MAP_CACHE
    if QUESTION_MAP_CACHE is not None:
        return QUESTION_MAP_CACHE
    session = SessionLocal()
    try:
        exact_maps = session.query(ReferenceQuestionMap).all()
        QUESTION_MAP_CACHE = [(m.pattern, m.standard_name) for m in exact_maps]
        return QUESTION_MAP_CACHE
    finally:
        session.close()


def call_llm_for_report(prompt):
    """
    OpenAI의 LLM을 호출하여 프롬프트에 대한 맞춤형 보고서 내용을 생성합니다.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 학생의 학습 성향 데이터를 분석하고 조언하는 전문 학습 코치입니다. 주어진 데이터를 기반으로, 학생에게 친절하고 지지적이지만, 전문적인 말투를 사용해 독창적인 보고서를 작성해 주세요.  T점수나 백분위 등의 표현을 지양하고 딱딱한 설명서가 아닌, 학생의 성장을 돕는 따뜻한 조언의 느낌을 담아주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.75,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"--- OpenAI API 호출 중 오류 발생: {e} ---")
        return f"--- [LLM 코멘트 생성 실패: {e}] ---"


def generate_report_with_llm(student_name: str, responses: dict, school_level: str = "초등"):
    """
    학생 데이터를 분석하고 LLM을 호출하여 맞춤형 보고서를 생성하고, 결과를 DB에 저장합니다.
    """
    db = SessionLocal()
    try:
        # --- 1. 데이터 로드 및 계산 ---
        # CSV 파일 로드 대신, Gradio 앱에서 직접 받은 responses 딕셔너리를 사용합니다.
        # 참조값은 DB에서 조회 (사용자가 선택한 학교급 기준)
        ref_level = school_level
        # 테이블/시드 보장 (배포 환경에서 테이블 미생성 대비)
        try:
            init_db()
            seed_reference_data()
        except Exception:
            pass
        std_info_df = get_std_info_df(ref_level)
        percentile_df = get_percentile_df()

        # 전달받은 responses로부터 항목별 원점수 집계 (DB의 질문→항목 매핑 사용)
        student_raw_scores = {}
        # 전수 매핑(정확 문항 텍스트)이 우선 (메모리 캐시)
        pattern_to_name = get_question_map_pairs()
        buckets = {}
        for q, val in responses.items():
            # 정확 일치 우선
            hit = False
            for pattern, name in pattern_to_name:
                if q == pattern:
                    buckets.setdefault(name, []).append(val)
                    hit = True
                    break
            if hit:
                continue
            # 안전장치: 포함 패턴 fallback (예외적으로)
            for pattern, name in pattern_to_name:
                if pattern in q:
                    buckets.setdefault(name, []).append(val)
                    hit = True
                    break
            # 미매핑 저장
            if not hit:
                try:
                    um_sess = SessionLocal()
                    try:
                        row = um_sess.query(ReferenceQuestionUnmapped).filter_by(question_text=q).first()
                        if row:
                            row.count = (row.count or 0) + 1
                            row.last_seen = datetime.now()
                        else:
                            um_sess.add(ReferenceQuestionUnmapped(question_text=q))
                        um_sess.commit()
                    finally:
                        um_sess.close()
                except Exception:
                    pass
        for name, vals in buckets.items():
            if len(vals) > 0:
                student_raw_scores[name] = float(sum(vals)) / len(vals) * 25  # 1~4척도 → 100점 환산 근사


        # 표준점수 계산
        student_scores = {}
        for std_name, raw_val in student_raw_scores.items():
            if std_name in std_info_df.index:
                mean = std_info_df.loc[std_name, '평균']
                std = std_info_df.loc[std_name, '표준편차']
                t_score = round(100 + 15 * ((raw_val - mean) / std))
                # T점수 범위 제한 및 백분위 조회
                if t_score < 0:
                    percentile = 0
                elif t_score > 200:
                    percentile = 99
                else:
                    percentile = percentile_df.loc[t_score, '백분위'] if t_score in percentile_df.index else "N/A"
                student_scores[std_name] = {
                    'raw': raw_val,
                    't_score': t_score,
                    'percentile': int(percentile) if percentile != "N/A" else 0
                }

        # 필수 항목 기본값 보정(동기 3종 + 전략/기술 구성요소 + 전략/기술 종합)
        required_list = [
            '자기성취', '사회적 관계', '직접적 보상처벌',
            '목표세우기', '계획하기', '실천하기', '돌아보기',
            '이해하기', '사고하기', '정리하기', '암기하기', '문제풀기',
            '학습전략', '학습기술',
        ]
        for required in required_list:
            if required not in student_scores and required in std_info_df.index:
                mean = std_info_df.loc[required, '평균']
                # 평균을 원점수로 간주하면 T=100이 되므로 백분위는 표에서 100이 없으면 50으로 처리
                t_score = 100
                percentile = percentile_df.loc[t_score, '백분위'] if t_score in percentile_df.index else 50
                student_scores[required] = {
                    'raw': mean,
                    't_score': t_score,
                    'percentile': int(percentile)
                }

        # 복합 지표(학습전략/학습기술) 보정: 구성 항목 평균으로 raw 근사 후 표준점수 계산
        def ensure_composite(composite_name: str, part_names: list[str]):
            if composite_name in student_scores:
                return
            available = [student_scores[p]['raw'] for p in part_names if p in student_scores]
            if len(available) == 0 or composite_name not in std_info_df.index:
                return
            raw_approx = float(sum(available)) / len(available)
            mean = std_info_df.loc[composite_name, '평균']
            std = std_info_df.loc[composite_name, '표준편차']
            t_score = round(100 + 15 * ((raw_approx - mean) / std))
            # T점수 범위 제한 및 백분위 조회
            if t_score < 0:
                percentile = 0
            elif t_score > 200:
                percentile = 99
            else:
                percentile = percentile_df.loc[t_score, '백분위'] if t_score in percentile_df.index else 50
            student_scores[composite_name] = {
                'raw': raw_approx,
                't_score': t_score,
                'percentile': int(percentile)
            }

        ensure_composite('학습전략', ['목표세우기', '계획하기', '실천하기', '돌아보기'])
        ensure_composite('학습기술', ['이해하기', '사고하기', '정리하기', '암기하기', '문제풀기'])

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
        def t(item: str) -> int:
            try:
                return int(student_scores[item]['t_score'])
            except Exception:
                return 100
        strategy_prompt = f"""
        '학습 전략/기술'에 대한 분석 및 코칭 코멘트를 작성해줘.

        [학생 데이터 요약]
        - 종합 분석: {s_analysis}
        - 학습 전략 종합 점수: T점수 {t('학습전략')} (백분위 {student_scores.get('학습전략', {}).get('percentile', 50)}%)
        - 학습 기술 종합 점수: T점수 {t('학습기술')} (백분위 {student_scores.get('학습기술', {}).get('percentile', 50)}%)
        - 강점 항목: 사고하기 (T={t('사고하기')})
        - 약점 항목: 목표세우기 (T={t('목표세우기')}), 이해하기 (T={t('이해하기')}), 문제풀기 (T={t('문제풀기')})

        [참고 가이드라인]
        - 핵심 특징: {s_analysis}
        - 코칭 방향: "{s_coaching_title}" 이 제목에 어울리는 내용으로, 전략(목표/계획) 보완과 기술(이해/문제풀이) 강화를 조언해줘.

        [작성 지침]
        - 강점(사고력)을 인정해주고, 약점(전략)을 보완하면 더 크게 성장할 수 있다는 점을 강조해줘.
        - 구체적인 활동 예시를 들어 조언해줘.
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

        # --- 3. 점수 테이블 생성 (시각적 개선) ---
        score_table_md = "### 📊 **학습 성향 측정 결과**\n\n"
        score_table_md += "> 전국 학생 데이터와 비교한 표준점수(T점수)와 백분위 결과입니다.\n\n"
        score_table_md += "| 🎯 구분 | 📋 영역 | 📈 원점수 | 🎯 표준점수(T) | 📊 백분위(%) |\n"
        score_table_md += "| :---: | :---: | :---: | :---: | :---: |\n"
        categories_in_order = {
            "💪 학습 동기": ['직접적 보상처벌', '사회적 관계', '자기성취'],
            "🎯 학습 전략": ['목표세우기', '계획하기', '실천하기', '돌아보기', '학습전략'],
            "🧠 학습 기술": ['이해하기', '사고하기', '정리하기', '암기하기', '문제풀기', '학습기술'],
            "😰 방해요인(심리)": ['스트레스민감성', '학습효능감', '친구관계', '가정환경', '학교환경'],
            "📱 방해요인(행동)": ['수면조절', '학습집중력', 'TV프로그램', '컴퓨터', '스마트기기']
        }
        
        for group, items in categories_in_order.items():
            for idx, item in enumerate(items):
                if item in student_scores:
                    score_data = student_scores[item]
                    item_name = '**종합**' if item in ['학습전략', '학습기술'] else item.replace('세우기','').replace('하기','')
                    
                    # 백분위에 따른 시각적 표시
                    percentile = score_data['percentile']
                    if percentile >= 84:
                        level_icon = "🔥"
                    elif percentile >= 50:
                        level_icon = "✅"
                    else:
                        level_icon = "⚠️"
                    
                    group_name = group if idx == 0 else ""  # 첫 번째 항목에만 그룹명 표시
                    score_table_md += f"| {group_name} | {item_name} | {score_data['raw']:.1f} | **{score_data['t_score']}** | {level_icon} **{percentile}%** |\n"

        # 점수 테이블 범례 추가
        score_table_md += "\n**📌 백분위 해석 가이드**\n"
        score_table_md += "- 🔥 **84% 이상**: 상위 16% (매우 우수)\n"
        score_table_md += "- ✅ **50% 이상**: 평균 이상 (양호)\n"
        score_table_md += "- ⚠️ **50% 미만**: 평균 이하 (개선 필요)\n\n"

        # 3-2. 종합 요약 프롬프트 (전반적인 경향성만 설명)
        summary_prompt = f"""
        학생의 전반적인 학습 성향을 종합하여 간결한 요약문을 작성해줘.
        
        [학생 데이터 요약]
        - 학생 이름: {student_name}
        - 주요 동기 유형: {m_type}
        - 학습 전략/기술 분석: {s_analysis}
        - 학습 방해 요인 분석: {h_analysis}
        - 구체적인 점수나 수치는 언급하지 말고, 전반적인 경향성만 설명해줘 : {score_table_md}
        
        [작성 지침]
        - 학생의 학습 성향을 10문장내로 간결하게 요약해줘.
        - 학생의 강점과 개선이 필요한 부분을 균형있게 언급해줘.
        - 따뜻하고 격려적인 톤으로 작성해줘.
        
        [출력 형식]
        학생의 학습 성향을 간결하게 요약한 8-10문장의 문단 (제목이나 서식 없이 본문만)
        """
        summary_comment = call_llm_for_report(summary_prompt)

        # --- 4. 최종 보고서 텍스트 (가독성 향상된 마크다운)
        report_md = f"""# 📊 {student_name} 학생 학습 성향 분석 종합 보고서

---

## 🎯 Ⅰ. 검사 결과 요약

> **{student_name}** 학생의 학습 성향을 종합적으로 분석한 결과입니다.

{summary_comment}

---

## 🚀 Ⅱ. 학습 성향 종합 분석 및 맞춤 코칭

### 💡 **1. 학습 동기** - 무엇이 나의 공부를 이끌고 있는가?

> **현재 동기 유형**: `{m_type}`

**📋 분석 결과**
{motivation_comment}

---

### 🎯 **2. 학습 전략/기술** - 나는 어떻게 공부하고 있는가?

> **현재 학습 수준**: `{s_analysis}`

**📋 분석 결과**  
{strategy_comment}

---

### ⚠️ **3. 학습 방해 요인** - 내 공부를 막는 것은 없는가?

> **방해 요인 분석**: `{h_analysis}`

**📋 분석 결과**  
{hindrance_comment}

---

## 🎉 마무리

이 보고서는 **{student_name}** 학생의 현재 학습 성향을 객관적으로 분석한 결과입니다. 
분석 결과를 바탕으로 자신의 강점을 더욱 발전시키고, 개선이 필요한 부분은 체계적으로 보완해 나가시기 바랍니다.

**💪 Remember**: 모든 학습자는 고유한 특성을 가지고 있으며, 자신만의 방식으로 성장할 수 있습니다!

---

## 🔍 디버깅용 - 상세 분석 결과

{score_table_md}

---
*📅 생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}*
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
