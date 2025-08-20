import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from dotenv import load_dotenv

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

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("데이터베이스 테이블이 생성되었습니다.")
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")

if __name__ == "__main__":
    init_db()
