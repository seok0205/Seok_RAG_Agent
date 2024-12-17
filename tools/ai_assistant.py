import os
from dotenv import load_dotenv
import tools.rag_hybrid_search as rhs
# import rag_hybrid_search as rhs
import tools.youtube_fuc as yf
# import youtube_fuc as yf

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from typing import Literal
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class AIAssistantConfig:
    """
    AI Assistant 설정 관리 데이터 클래스
    시스템 구성에 필요한 설정값 한 군데에서 관리
    """

    llm_model: str
    openai_api_key: str
    embedding_model: str
    youtube_api_key: str
    vector_store_path: str
    processed_docs_path: str
    temperature: float = 0.0

    not_supported_message: str = "죄송함다. AI 관련 질문에 대해서만 답변 가능합니다."

class AgentAction(BaseModel):
    """
    에이전트 행동 정의 Pydantic 모델
    행동의 타입과 입력을 구조화하여 관리
    """
    action: Literal["search_video", "search_news", "not_supported"]
    action_input: str
    search_keyword: str = ""

class AIAssistant:
    """
    youtube 검색, 뉴스 검색 기능 통합 assistant
    rag와 youtue 검색 결합, 정보 검색 제공
    """

    @classmethod
    def from_env(cls) -> "AIAssistant":
        """
        환경 변수파일에서 설정값 로드, AIAssistant 인스턴스 생성하는 클래스 메서드
        """
        load_dotenv(dotenv_path='seok25.env')

        config = AIAssistantConfig(
            llm_model=os.getenv("OPENAI_MODEL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL"),
            youtube_api_key=os.getenv("YT_API_KEY"),
            vector_store_path=os.getenv("VECTOR_STORE_NAME"),
            processed_docs_path=os.getenv("PROCESSED_DOCS_PATH"),
            temperature=0.0,
        )
        return cls(config)
    
    def __init__(self, config: AIAssistantConfig):
        """
        AI Assistant 컴포넌트 설정 및 초기화
        """
        self.config = config

        self.llm = ChatOpenAI(temperature=config.temperature, model=config.llm_model, api_key=config.openai_api_key)

        self.output_parser = JsonOutputParser(pydantic_object=AgentAction)

        self.youtube_tool = yf.YouTubeSearchTool(config.youtube_api_key)

        self.embedding_model = OpenAIEmbeddings(model=config.embedding_model, api_key=config.openai_api_key)
        self.rag = rhs.AINewsRAG(self.embedding_model)

        self.rag.load_vector_store(config.vector_store_path, config.processed_docs_path)

        self.tools = [
            Tool(
                name="search_video",
                func=self.youtube_tool.search_videos,
                description="AI 관련 YouTube 영상을 검색합니다.",
            ),
            Tool(
                name="search_news",
                func=self.rag.hybrid_search,
                description="AI 관련 뉴스 기사를 검색합니다."
            )
        ]

        self.prompt = PromptTemplate(
            input_variables=["input"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
            template=
            """
            당신은 AI 관련 정보를 제공하는 도우미입니다.
            먼저 입력된 질의가 AI 관련 내용인지 확인하세요.
            
            AI 관련 주제 판단 기준:
            - AI 기술 (머신러닝, 딥러닝, 자연어처리 등)
            - AI 도구 및 서비스 (ChatGPT, DALL-E, Stable Diffusion 등)
            - AI 회사 및 연구소 소식
            - AI 정책 및 규제
            - AI 교육 및 학습
            - AI 윤리 및 영향

            AI 관련 질의가 아닌 경우:
            - action을 "not_supported"로 설정
            - search_keyword는 빈 문자열로 설정

            AI 관련 질의인 경우 다음 작업을 수행하세요:
            1. 검색 도구 설정: 질의 의도 분석 기반 최적 도구 선택하여 action의 값으로 대입
            2. 키워드 추출 : 최적화 검색어 생성 후 search_keyword의 값으로 대입
            3. 질의 내용은 : action_input의 값으로 대입

            사용 가능한 도구:
            1. search_video : AI 관련 영상 콘텐츠 검색 특화
            2. search_news : AI 관련 뉴스 및 기사 검색 특화

            도구 선택 기준:
            A) search_video 선정 조건:
                - 영상 콘텐츠 요구 (영상, 동영상)
                - 교육 자료 요청 (강의, 강좌, 수업)
                - 실습 가이드 (튜토리얼, 가이드, 설명)
                - 시각적 설명 (시연, 데모)

            B) search_news 선정 조건:
                - 뉴스 콘텐츠 (뉴스, 소식)
                - 기사 요청 (기사, 글)
                - 정보 탐색 (정보, 현황, 동향)
                - 연구 자료 (연구, 조사, 분석)
            
            키워드 추출 규칙:
            1. 핵심 주제어 분리
                - AI 관련 핵심 개념 추출
                - 매체 유형 지시어 제거 (정보, 뉴스, 영상, 기사 등)
                - 보조어 및 조사 제거
            
            2. 의미론적 최적화
                - 전문 용어 완전성 유지
                - 개념 간 관계성 보존
                - 맥락 적합성 확보

            분석 대상 질의: {input}

            {format_instructions}
            """,
        )

        self.chain = RunnableSequence(
            first=self.prompt,
            middle=[self.llm],
            last=self.output_parser,
        )

    def process_query(self, query: str) -> str:
        """
        사용자 질문 처리, 적절한 도구 사용해 결과 반환하는 메서드
        """
        try:
            result = self.chain.invoke({"input": query})

            # 결과에서 필요한 정보 추출
            action = result["action"]   # 선택 도구
            action_input = result["action_input"]   # 원본 입력
            search_keyword = result["search_keyword"]   # 추출된 검색어

            # 디버깅용 정보 출력
            print(f"\n검색어 분석 결과:")
            print(f"- 원본 질문: {action_input}")
            print(f"- 추출된 키워드: {search_keyword}")
            print(f"- 선택된 도구: {action}\n")

            # ai 관련 쿼리가 아니면
            if action == "not_supported":
                return {'action': action, 'response': self.config.not_supported_message}
            
            for tool in self.tools:
                if tool.name == action:
                    return {'action': action, 'response': tool.func(search_keyword)}
                
            return f"지원 불가: {action}"
        
        except Exception as e:
            return f"처리 중 오류 발생: {e}"
        
def main():
    """
    메인 함수 및 시스템 테스트(테스트 케이스)
    """
    assistant = AIAssistant.from_env()
    print("\nAI 정보 검색 도우미를 시작합니다.")

    test_cases = [
        "AI 관련 기사 찾아줘",
        "머신러닝 입문 강의 추천해주세요.",
    ]

    for query in test_cases:
        print(f"질문: {query}")
        result = assistant.process_query(query)
        print(f"답변: {result}\n")
    

if __name__ == "__main__":
    main()
