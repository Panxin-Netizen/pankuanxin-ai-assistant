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
        # 1. 加载文档 (增加 encoding='utf-8' 防止中文乱码)
        loader = DirectoryLoader('docs', glob="**/*.txt", loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
        documents = loader.load()

        if not documents:
            st.error("❌ docs 文件夹下没有找到 txt 文件，请检查路径！")
            return None

        # 2. 文本切分
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 3. 向量化 (模型会自动从缓存或镜像源加载)
        embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

        # 4. 存入向量数据库 (persist_directory 指定保存路径，实现持久化)
        # 注意：如果是第一次运行，这会创建 ./chroma_db 文件夹
        db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
        st.success(f"✅ 知识库构建完成！共加载 {len(documents)} 个文档，切分为 {len(texts)} 个片段。")
        return db

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