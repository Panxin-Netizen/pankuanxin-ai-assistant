import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="个人专属AI助手", page_icon="🤖")
st.title("🤖 潘宽心的AI知识库问答助手")

# --- 2. 侧边栏配置 (安全输入 API Key) ---
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("请输入 DeepSeek API Key:", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
        st.success("Key 已设置！")
    else:
        st.warning("⚠️ 请先在左侧输入 API Key 才能开始对话")

# --- 3. 核心逻辑：加载与构建知识库 ---
@st.cache_resource
def load_knowledge_base():
    """
    加载文档并向量化。
    使用 cache_resource 确保只在第一次运行时执行，后续直接复用内存中的对象。
    """
    with st.spinner("🧠 正在初始化知识库，请稍候..."):
        # 1. 获取当前脚本所在的绝对路径 (解决云端路径问题)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # ⚠️ 注意：请把 'resume.txt' 换成你在 GitHub 上真实的文件名！
        file_name = "resume.txt"
        file_path = os.path.join(current_dir, file_name)

        # 2. 检查文件是否存在 (防止报错)
        if not os.path.exists(file_path):
            st.error(f"❌ 找不到文件: {file_name}。请检查文件名是否正确，或是否已上传到 GitHub。")
            return None

        # 3. 使用 TextLoader 直接加载该文件
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        if not documents:
            st.error("❌ 文件内容为空，无法构建知识库！")
            return None

        # 4. 文本切分 (保持原有逻辑)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 5. 向量化 (保持原有逻辑)
        # embeddings = ... (你原本的向量化代码)
        # vectorstore = ... (你原本的向量数据库代码)

        # 返回结果 (根据你的后续代码调整返回值)
        return texts  # 或者返回 vectorstore

# --- 4. 启动应用 ---
# 只有当用户输入了 Key 之后，才开始加载模型和数据库
if 'qa_chain' not in st.session_state and api_key:
    db = load_knowledge_base()

    if db:
        # 配置大模型
        llm = ChatOpenAI(
            model_name="deepseek-chat",
            temperature=0.5
        )

        # 创建检索问答链
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3})
        )

# --- 5. 聊天界面 ---
user_input = st.chat_input("请输入关于潘宽心的问题...")

if user_input:
    if not api_key:
        st.error("请先在左侧侧边栏输入 API Key！")
    elif 'qa_chain' not in st.session_state:
        st.error("知识库正在加载中，请稍后再试...")
    else:
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            try:
                # 显示思考状态
                with st.spinner("正在思考..."):
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                st.write(response['result'])
            except Exception as e:
                st.error(f"出错了：{e}")
                st.info("提示：请检查 API Key 是否正确，或网络连接是否正常。")
