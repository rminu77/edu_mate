import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./learning_assessment.db")

Base = declarative_base()

class SurveyResponse(Base):
    __tablename__ = "survey_responses"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    student_name = Column(String, index=True, nullable=True)
    responses_json = Column(Text, nullable=False)
    scores_json = Column(Text, nullable=True)
    report_content = Column(Text, nullable=True)

class LLMLog(Base):
    __tablename__ = "llm_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    interaction_type = Column(String, nullable=False)
    input_data = Column(Text, nullable=False)
    output_data = Column(Text, nullable=False)

# 참조 데이터 (표준점수 평균/표준편차)
class ReferenceStandard(Base):
    __tablename__ = "reference_standards"
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String, index=True, nullable=False)  # 초등/중등/고등
    name = Column(String, index=True, nullable=False)   # 항목명 (예: 자기성취)
    mean = Column(Float, nullable=False)
    std = Column(Float, nullable=False)

# 참조 데이터 (T점수 → 백분위)
class ReferencePercentile(Base):
    __tablename__ = "reference_percentiles"
    id = Column(Integer, primary_key=True, index=True)
    t_score = Column(Integer, index=True, nullable=False)
    percentile = Column(Integer, nullable=False)

# 질문 → 항목 매핑 (패턴 기반)
class ReferenceQuestionMap(Base):
    __tablename__ = "reference_question_map"
    id = Column(Integer, primary_key=True, index=True)
    pattern = Column(String, index=True, nullable=False)        # 질문 텍스트에 포함될 키워드/패턴
    standard_name = Column(String, index=True, nullable=False)  # 표준 항목명(예: 사회적 관계)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("데이터베이스 테이블이 생성되었습니다.")
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")


def seed_reference_data():
    """
    프로젝트의 refer/ 폴더에 있는 참조 CSV들을 읽어 DB에 저장합니다.
    - 표준점수 - 초등.csv, 표준점수 - 중등.csv, 표준점수 - 고등.csv
      (index: 항목명, columns: 평균, 표준편차)
    - 백분위점수.csv (columns: 표준점수, 백분위)
    이미 데이터가 존재하면 시드를 건너뜁니다.
    """
    session = SessionLocal()
    try:
        # 이미 시드되었는지 확인
        has_std = session.query(ReferenceStandard).first() is not None
        has_pct = session.query(ReferencePercentile).first() is not None
        has_qmap = session.query(ReferenceQuestionMap).first() is not None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        refer_dir = os.path.join(base_dir, "refer")
        if not os.path.isdir(refer_dir):
            print(f"참조 디렉토리를 찾을 수 없습니다: {refer_dir}")
            return

        if not has_std:
            level_files = {
                "초등": os.path.join(refer_dir, "표준점수 - 초등.csv"),
                "중등": os.path.join(refer_dir, "표준점수 - 중등.csv"),
                "고등": os.path.join(refer_dir, "표준점수 - 고등.csv"),
            }
            for level, path in level_files.items():
                if os.path.isfile(path):
                    try:
                        df = pd.read_csv(path, index_col=0)
                        # 기대 컬럼명: '평균', '표준편차'
                        for name, row in df.iterrows():
                            mean_val = float(row.get("평균"))
                            std_val = float(row.get("표준편차"))
                            session.add(ReferenceStandard(level=level, name=str(name), mean=mean_val, std=std_val))
                        session.commit()
                        print(f"표준점수 시드 완료: {level}")
                    except Exception as e:
                        session.rollback()
                        print(f"표준점수 시드 중 오류({level}): {e}")
                else:
                    print(f"파일 없음(표준점수): {path}")

        if not has_pct:
            pct_path = os.path.join(refer_dir, "백분위점수.csv")
            if os.path.isfile(pct_path):
                try:
                    df = pd.read_csv(pct_path)
                    # 기대 컬럼명: '표준점수', '백분위'
                    # 일부 파일은 index로 설정되어 있을 수 있으므로 보정
                    if "표준점수" not in df.columns and df.shape[1] >= 2:
                        df.columns = ["표준점수", "백분위"]
                    for _, row in df.iterrows():
                        t = int(row.get("표준점수"))
                        p = int(row.get("백분위"))
                        session.add(ReferencePercentile(t_score=t, percentile=p))
                    session.commit()
                    print("백분위점수 시드 완료")
                except Exception as e:
                    session.rollback()
                    print(f"백분위점수 시드 중 오류: {e}")
            else:
                print(f"파일 없음(백분위점수): {pct_path}")

        # 기본 질문→항목 매핑 시드 (패턴 기반, 필요시 대시보드/SQL로 가감)
        if not has_qmap:
            default_map = {
                # 전략/기술
                "목표": "목표세우기", "계획": "계획하기", "실천": "실천하기", "돌아보": "돌아보기",
                "이해": "이해하기", "사고": "사고하기", "정리": "정리하기", "암기": "암기하기", "문제": "문제풀기",
                # 방해
                "스트레스": "스트레스민감성", "효능": "학습효능감", "친구": "친구관계", "가정": "가정환경",
                "학교": "학교환경", "수면": "수면조절", "집중": "학습집중력", "TV": "TV프로그램",
                "컴퓨터": "컴퓨터", "스마트": "스마트기기",
                # 동기
                "보상": "직접적 보상처벌", "관계": "사회적 관계", "성취": "자기성취",
            }
            try:
                for pattern, name in default_map.items():
                    session.add(ReferenceQuestionMap(pattern=pattern, standard_name=name))
                session.commit()
                print("질문→항목 매핑 시드 완료")
            except Exception as e:
                session.rollback()
                print(f"질문 매핑 시드 중 오류: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    init_db()
    seed_reference_data()
