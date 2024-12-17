# Agent

- AI 관련 뉴스 및 Youtube AI 내용 검색 Agent.
- AI에 관련된 질문을 챗봇에 제시하면 그에 대한 알맞은 기사 및 동영상 5가지를 제시해줌.

## 실행 방법(Windows)

1. 가상환경 만들기 : `python -m venv [가상환경이름]`

2. 가상환경 활성화 : `venv\Scripts\activate` or `venv\Scripts\activate.bat`

    - Mac/Linux : `source venv/bin/activate`

3. 필요 패키지 설치 : `pip install -r requirements.txt`

4. 환경 설정 : 프로젝트 폴더 내에 `.env` 파일 만들고 아래 내용 입력

```env
YT_API_KEY='본인의 유튜브 api-key'

OPENAI_API_KEY='본인의 api-key'

OPENAI_MODEL="gpt-4o-mini"

OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"

VECTOR_STORE_NAME="ai_news_vectorstore"

PROCESSED_DOCS_PATH="processed_docs/processed_docs.pkl"

NEWS_FILE_PATH="./ai_news"
```

## 파일 설명

1. [ai_news](./ai_news/) : AI times에서 크롤링한 기사 모음(2020~2024년)

2. [tools](./tools/) : Agent 기능 구현에 필요한 파일들

    - [ai_assistant.py](./tools/ai_assistant.py)
        - `AIAssistant()` : youtube 검색, 뉴스 검색 기능 통합 assistant, rag와 youtue 검색 결합, 정보 검색 제공

    - [collect_news_data.py](./tools/collect_news_data.py)
        - AI times에서 뉴스들을 크롤링하는 기능. `ai_news` 폴더에 저장됨.

    - [rag_hybrid_search.py](./tools/rag_hybrid_search.py)
        - 입력받은 질문으로 검색을 할 때, hybrid search, BM25 알고리즘 이용한 키워드검색을 추가하여 검색의 정확도를 높임. 의미 검색과 키워드 검색의 결합.

    - [youtube_fuc.py](./tools/youtube_fuc.py)
        - 유튜브 API를 활용한 유튜브 동영상을 검색하고, 상세 정보 중 필요한 정보(제목, 조회수, url 등)만 추출하여 가독성 좋게 정리.

3. [main.py](./main.py) : 위의 기능을 결함한 `AIAssistant()`를 Streamlit으로 구현.

### 추가 생성 파일

- 환경 설정 진행 시, 본인의 가상환경 파일 및 `.env` 파일 생성
- `main.py` 실행 시, `ai_news_vectorstore` 및 `processed_docs` 생성
- `__pycache__` 파일 생성

## 사용 기술

- BeautifulSoup
- FAISS
- BM25
- Langchain
- Hybrid search
- pydantic
- youtube api
