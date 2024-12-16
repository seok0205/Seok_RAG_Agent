import os
import json
import glob
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from tqdm import tqdm
import sys

class AINewsRAG:
    """
    AI news 검색 RAG 시스템:
    뉴스 기사를 벡터 DB로 변환, 의미론적 검색과 키워드 기반 검색을 결합한 하이브리드 검색 기능 제공
    """
    def __init__(self, embedding_model):
        """클래스 초기화"""
        self.embeddings = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.bm25 = None
        self.processed_docs = None
        self.doc_mapping = None
        
        # 로깅 설정
        self.logger = logging.getLogger('AINewsRAG')
        # 기존 핸들러 제거
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        # 로그 중복 방지
        self.logger.propagate = False
        
    def load_json_files(self, directory_path: str) -> List[Dict]:
        """기사가 담긴 JSON 파일들을 로드"""
        all_documents = []
        json_files = glob.glob(f"{directory_path}/ai_times_news_*.json")
        
        self.logger.info(f"총 {len(json_files)}개의 JSON 파일을 로드합니다...")
        
        for file_path in tqdm(json_files, desc="JSON 파일 로드 중"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents = json.load(file)
                    if documents :
                        documents = [doc for doc in documents if len(doc['content']) > 10]
                    #기사 내용이 없으면 생략
                    if len(documents) >= 10 : 
                        all_documents.extend(documents)
            except Exception as e:
                self.logger.error(f"파일 로드 중 오류 발생: {file_path} - {str(e)}")
        
        self.logger.info(f"총 {len(all_documents)}개의 뉴스 기사를 로드했습니다.")
        return all_documents
    
    def process_documents(self, documents: List[Dict]) -> List[Document]:
        """문서를 처리하고 청크로 분할합니다."""
        processed_docs = []
        self.logger.info("문서 처리 및 청크 분할을 시작합니다...")
        
        for idx, doc in enumerate(tqdm(documents, desc="문서 처리 중")):
            try:
                full_text = f"제목: {doc['title']}\n내용: {doc['content']}"
                metadata = {
                    'doc_id': idx,
                    'title': doc['title'],
                    'url': doc['url'],
                    'date': doc['date']
                }
                
                chunks = self.text_splitter.split_text(full_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            'chunk_id': f"doc_{idx}_chunk_{chunk_idx}"  # 청크별 고유 ID
                        }
                    ))
            except Exception as e:
                self.logger.error(f"문서 처리 중 오류 발생: {doc.get('title', 'Unknown')} - {str(e)}")
        
        self.processed_docs = processed_docs
        self.initialize_bm25(processed_docs)

        self.logger.info(f"총 {len(processed_docs)}개의 청크가 생성되었습니다.")
        
        return processed_docs
    
    def initialize_bm25(self, documents: List[Document]):
        """
        BM25 검색 엔진을 초기화
        """
        self.logger.info("BM25 검색 엔진을 초기화합니다...")
        
        tokenized_corpus = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_mapping = {
            i: doc for i, doc in enumerate(documents)
        }
        
        self.logger.info("BM25 검색 엔진 초기화가 완료되었습니다.")
    
    def create_vector_store(self, documents: List[Document]):
        """FAISS 벡터 스토어를 생성"""
        self.logger.info("벡터 스토어 생성을 시작합니다...")
        total_docs = len(documents)
        
        try:
            # 청크를 더 작은 배치로 나누어 처리
            batch_size = 100
            for i in tqdm(range(0, total_docs, batch_size), desc="벡터 생성 중"):
                batch = documents[i:i+batch_size]
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                    self.vector_store.merge_from(batch_vectorstore)
            
            self.logger.info("벡터 스토어 생성이 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
            raise

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        키워드 기반 BM25 검색을 수행
        """
        if self.bm25 is None:
            raise ValueError("BM25가 초기화되지 않았습니다.")
        
        self.logger.info(f"'{query}' 키워드 검색을 시작합니다...")
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_k_idx = np.argsort(bm25_scores)[-k:][::-1]
        results = [
            (self.doc_mapping[idx], bm25_scores[idx])
            for idx in top_k_idx
        ]
        
        self.logger.info(f"{len(results)}개의 키워드 검색 결과를 찾았습니다.")
        return results
    
    def hybrid_search(
            self, 
            query: str, 
            k: int = 5, 
            semantic_weight: float = 0.5
        ) -> List[Tuple[Document, float]]:
            """
            의미론적 검색과 키워드 검색을 결합한 하이브리드 검색을 수행
            """
            self.logger.info(f"'{query}' 하이브리드 검색을 시작합니다...")
            
            # 의미론적 검색 수행
            self.logger.info(f"'{query}' 의미론적 검색을 시작합니다...")
            semantic_results = self.vector_store.similarity_search_with_score(query, k=k)
            self.logger.info(f"{len(semantic_results)}개의 의미론적 검색 결과를 찾았습니다.")

            # 키워드 기반 검색 수행
            keyword_results = self.keyword_search(query, k=k)
            
            # 문서 ID를 키로 사용
            combined_scores = {}
            
            # 의미론적 검색 결과 처리
            max_semantic_score = max(score for _, score in semantic_results)
            for doc, score in semantic_results:
                doc_id = doc.metadata['chunk_id']
                
                #5개의 문서의 점수가
                normalized_score = 1 - (score / max_semantic_score) 
                combined_scores[doc_id] = {
                    'doc': doc,
                    'score': semantic_weight * normalized_score
                }
            
            # 키워드 검색 결과 처리
            max_keyword_score = max(score for _, score in keyword_results)
            for doc, score in keyword_results:
                doc_id = doc.metadata['chunk_id']
                normalized_score = score / max_keyword_score
                if doc_id in combined_scores:
                    combined_scores[doc_id]['score'] += (1 - semantic_weight) * normalized_score
                else:
                    combined_scores[doc_id] = {
                        'doc': doc,
                        'score': (1 - semantic_weight) * normalized_score
                    }
            
            # 결과 정렬
            sorted_results = sorted(
                [(info['doc'], info['score']) for info in combined_scores.values()],
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            self.logger.info(f"{len(sorted_results)}개의 하이브리드 검색 결과를 찾았습니다.")
            return sorted_results

    def save_vector_store(self, vector_store_path: str, processed_docs_path:str=None):
        """
        벡터 스토어와 BM25 데이터를 저장
        """
        try:
            self.logger.info(f"데이터를 {vector_store_path}에 저장합니다...")
            
            # 벡터 스토어 저장
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)
            
            # processed_docs 저장
            if self.processed_docs:
                os.makedirs(os.path.dirname(processed_docs_path), exist_ok=True)
                with open(processed_docs_path, 'wb') as f:
                    pickle.dump(self.processed_docs, f)
            
            self.logger.info("저장이 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"저장 중 오류 발생: {str(e)}")
            raise

    
    def load_vector_store(self, vector_store_path: str, processed_docs_path):
        """
        벡터 스토어와 BM25 데이터를 로드
        """
        try:
            self.logger.info(f"데이터를 {vector_store_path}에서 로드합니다...")
            
            # 벡터 스토어 로드
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # processed_docs 로드
            if os.path.exists(processed_docs_path):
                with open(processed_docs_path, 'rb') as f:
                    self.processed_docs = pickle.load(f)
                self.initialize_bm25(self.processed_docs)
            
            self.logger.info("로드가 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"로드 중 오류 발생: {str(e)}")
            raise

load_dotenv(dotenv_path='seok.env')

# 임베딩 모델 초기화 
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
)

# 환경 변수에서 경로 가져오기
vector_store_path = os.getenv("VECTOR_STORE_NAME", "ai_news_vectorstore")
news_dir = os.getenv("NEWS_FILE_PATH", "./ai_news")
processed_doc_path = os.getenv("PROCESSED_DOCS_PATH", "processed_docs/processed_docs.pkl")

# RAG 시스템 초기화
rag = AINewsRAG(embedding_model)

print("새로운 벡터 스토어를 생성합니다...")

# JSON 파일에서 뉴스 데이터 로드
documents = rag.load_json_files(news_dir)

# 문서 처리 및 벡터 스토어 생성
processed_docs = rag.process_documents(documents)
rag.create_vector_store(processed_docs)

# 벡터 스토어 저장
rag.save_vector_store(vector_store_path, processed_doc_path)
print("새로운 벡터 스토어 생성 및 저장 완료")