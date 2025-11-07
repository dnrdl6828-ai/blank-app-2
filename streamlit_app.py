import os
import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    # 1. Tavily Search Tool 호출하기    
    return TavilySearchResults(k=6, name="web_search")


# --------------------------------------------------------------------
# 2. PDF Tool
# --------------------------------------------------------------------
def load_pdf_files(uploaded_files):
    # 2. PDF 로더 초기화 및 문서 불러오기    
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 3. 텍스트를 일정 단위(chunk)로 분할하기
    #    - chunk_size: 한 덩어리의 최대 길이
    #    - chunk_overlap: 덩어리 간 겹치는 부분 길이

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    split_docs = text_splitter.split_documents(all_documents)

    # 4. 분할된 문서들을 임베딩하여 벡터 DB(FAISS)에 저장하기

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

    # 5. 검색기(retriever) 객체 생성

    retriever = vector.as_retriever(search_kwargs={"k": 5})

    # 6. retriever를 LangChain Tool 형태로 변환 -> name은 pdf_search로 지정    

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="This tool gives you direct access to the uploaded PDF documents. "
                    "Always use this tool first when the question might be answered from the PDFs."
    )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        # 7. 여러분의 챗봇에 맞는 system message 작성하기         
         "You are a helpful assistant for KIBO employees. "
         "First, always try `pdf_search`. "
         "If `pdf_search` returns no relevant results, immediately call ONLY `web_search`. "
         "Never mix the two tools. "
         "Answer in Korean with a professional and friendly tone, including emojis."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 8.agent 및 aagent_executor 생성하기

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수 (툴 사용 내역 제거)
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]

    # 9. intermediate_steps 통해 사용툴을 출력할 수 있는 코드 완성하기

    if result.get("intermediate_steps"):
        last_action, _ = result["intermediate_steps"][-1]
        answer += f"\n\n출처:\n- Tool: {last_action.tool}, Query: {last_action.tool_input}"

    return f"답변:\n{answer}"


# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():

    # 10. 여러분의 챗봇에 맞는 스타일로 변경하기

    st.set_page_config(page_title="「차세대 챗봇 시스템 기반 지식공유 플랫폼」", layout="wide", page_icon="🤖")
    #st.image('data/kibo_image.jpg', width=800)
    st.image('image/AI.jpg', width=800)
    st.markdown('---')
    st.title("「차세대 챗봇 시스템 기반 지식공유 플랫폼」")   

    with st.sidebar:
        openai_api = st.text_input("OPENAI API 키", type="password")
        tavily_api = st.text_input("TAVILY API 키", type="password")
        pdf_docs = st.file_uploader("PDF 파일 업로드", accept_multiple_files=True)

    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        tools = [search_web()]
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))

        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("질문을 입력하세요")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키를 입력하세요.")


if __name__ == "__main__":
    main()
