from datetime import datetime
import requests

class YouTubeSearchTool:
    """youtube 검색 수행 도구 클래스"""
    def __init__(self, api_key: str):
        """youtube api client 초기화"""
        if not api_key:
            raise ValueError("Youtube API 키가 필요합니다.")
        
        self.api_key = api_key
        self.search_url = "https://www.googleapis.com/youtube/v3/search"
        self.video_url = "https://www.googleapis.com/youtube/v3/videos"

    def search_videos(self, query: str, max_results: int = 5) -> str:
        """
        youtube에서 비디오 검색, 상세 정보 반환
        """
        try:
            # API 호출 설정 파라미터
            params = {
                'key': self.api_key,
                'q': query,
                'part': 'snippet',
                'videoType': 'any',
                'maxResults': max_results,
                'type': 'video',
                'order': 'relevance',
                'regionCode': 'KR',
                'relevanceLanguage': 'ko'
            }

            # API 요청 실행
            response = requests.get(self.search_url, params=params)
            response.raise_for_status() # HTTP 에러 체크

            # 검색 결과 처리
            search_data = response.json()
            if 'items' not in search_data or not search_data['items']:
                return "검색 결과가 없습니다."
            
            video_list = []

            # 각 비디오 상세 정보 수집
            for item in search_data['items']:
                try:
                    video_id = item.get('id', {}).get('videoId')
                    # 비디오 통계 정보 조회(조회수 및 좋아요)
                    video_stats = self._get_video_stats(video_id)

                    # 날짜 형식 변환 (ISO -> 사용자 친화적 형식)
                    published_at = datetime.strptime(
                        item['snippet']['publishedAt'],
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    formatted_date = published_at.strftime("%Y년 %m월 %d일")

                    # 비디오 정보 구조화
                    video = {
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'publishedTime': formatted_date,
                        'description': item['snippet']['description'],
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'view_count': int(video_stats.get('viewCount', 0)),
                        'like_count': int(video_stats.get('likeCount', 0)),
                    }
                    video_list.append(video)
                except Exception as e:
                    print(f"비디오 정보 처리 중 오류 발생: {e}")
                    continue

            if not video_list:
                return "검색된 영상의 상세 정보를 수집하는데 실패했습니다."
            
            # 결과 포맷팅 및 반환
            return self._format_results(query, video_list)
        
        except requests.exceptions.RequestException as e:
            return f"youtube api 호출 중 오류 발생: {e}"
        except Exception as e:
            return f"검색 중 오류 발생: {e}"
        
    def _get_video_stats(self, video_id: str) -> dict:
        """
        개별 비디오의 통계 정보 조회하는 private 메서드
        """
        try:
            # API 요청 매개변수 설정
            params = {
                'key': self.api_key,
                'id': video_id,
                'part': 'statistics' # 통계 정보만 요청
            }

            # API 호출 및 결과 처리
            response = requests.get(self.video_url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get('items'):
                return data['items'][0]['statistics']
            return {}
        
        except Exception as e:
            print(f"비디오 통계 정보 조회 중 오류 발생: {e}")
            return {}
        
    def _format_results(self, query: str, videos: list) -> str:
        """
        검색 결과를 사용자 친화적인 문자열로 포맷팅하는 private 메서드
        """
        # 검색 결과 헤더 작성
        result = f"'{query}'에 대한 검색 결과:\n"
        result += f"총 {len(videos)}개의 영상이 검색됨\n\n"

        # 각 비디오 정보 포맷팅
        for i, video in enumerate(videos, 1):
            result += f"{i}. {video['title']}\n"
            result += f"    채널: {video['channel']}\n"
            result += f"    날짜: {video['publishedTime']}\n"
            result += f"    조회수: {video['view_count']:,}회\n"
            result += f"    좋아요: {video['like_count']:,}명\n"
            result += f"    URL: {video['url']}\n"
            result += f"    설명: {video['description'][:150]}...\n\n"

        return result