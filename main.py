import os
import streamlit as st
from openai import OpenAI
from tools.ai_assistant import AIAssistant
from dotenv import load_dotenv

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
if "messages" not in st.session_state:  # 입력값에 대한 메시지
    st.session_state["messages"] = []
if "agent" not in st.session_state:  # 입력값에 대한 메시지
    st.session_state["agent"] = AIAssistant.from_env()

load_dotenv(dotenv_path='seok25.env')
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.markdown(
    """
    <style> 
        .stMainBlockContainer {
            max-width: none;
            margin: 0 auto;
            width: 75vw;
            position: relative;
        }
        .video_info_wrap {
            display: flex;
            gap: 30px;
        }
        .video_info .video_info_title {
            font-weight: bold;
            font-size: 15px;
        }
        .video_title span {
            font-weight: bold;
            font-size: 20px;
        }
        .stChatMessageAvatarAssistant+.stChatMessageContent .stVerticalBlock {
            color:red;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def display_video(video_list):
    """YouTube 비디오 정보를 표시하는 함수"""
    with st.container():
        for video in video_list:
            st.markdown(
                f"""
                <div class="video_title">
                    <span>{video["title"]}</span>
                </div>
                <div class="video_info_wrap">
                    <div class="video_iframe">
                        <iframe width="640" height="360" src="{video["url"].replace("watch?v=", "embed/")}" 
                        title="{video["title"]}" frameborder="0" allow="accelerometer; autoplay; 
                        clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                    </div>
                    <div class="video_info">
                        <div>
                            <div class="video_info_title"><span>채널</span></div>
                            <div><span>{video["channel"]}</span></div>
                        </div>
                        <div>
                            <div class="video_info_title"><span>게시일</span></div>
                            <div><span>{video["publishTime"]}</span></div>
                        </div>
                        <div>
                            <div class="video_info_title"><span>URL</span></div>
                            <div><span>{video["url"]}</span></div>
                        </div>
                        <div>
                            <div class="video_info_title"><span>조회수</span></div>
                            <div><span>👀 {video["view_count"]}회</span></div>
                        </div>
                        <div>
                            <div class="video_info_title"><span>좋아요 수</span></div>
                            <div><span>❤ {video["like_count"]}개</span></div>
                        </div>
                        <div>
                            <div class="video_info_title"><span>설명</span></div>
                            <div><span>{video["description"]}</span></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.title("🤖 Agent 채팅봇")
st.write("AI 비디오 및 텍스트 정보를 제공하는 챗봇입니다. 원하는 메시지를 입력해주세요!")

# 사용자 입력 받기
query = st.chat_input("메시지를 입력해주세요")
if query:
    # 기존 메시지 표시
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):  # 비디오 리스트 출력
                display_video(message["content"])
            else:
                st.write(message["content"])

    # 사용자 입력 메시지 기록
    user_message = {"role": "user", "content": query}
    st.session_state["messages"].append(user_message)

    with st.chat_message("user"):
        st.write(query)

    # 어시스턴트 처리 및 출력
    with st.chat_message("assistant"):
        response = st.session_state["agent"].process_query(query)
        if isinstance(response, list):  # 비디오 결과 처리
            display_video(response)
        else:
            st.write(response)

        # 어시스턴트 메시지 기록
        st.session_state["messages"].append({"role": "assistant", "content": response})