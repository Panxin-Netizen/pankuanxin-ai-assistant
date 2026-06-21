import streamlit as st
# --- 1. 导入必要的库 ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

# --- 2. 页面配置 ---
st.set_page_config(page_title="个人专属AI助手", page_icon="🤖")
st.title("🤖 潘宽心的AI知识库问答助手")

# --- 3. 侧边栏配置 (安全输入 API Key) ---
with st.sidebar:
    st.header("⚙️ 设置")
    # 获取用户输入的 API Key
    api_key = st.text_input("请输入 DeepSeek API Key:", type="password", placeholder="sk-...")
    if api_key:
        st.success("Key 已设置！")
    else:
        st.warning("⚠️ 请先在左侧输入 API Key 才能开始对话")

# --- 4. 核心逻辑：加载与构建知识库 ---
@st.cache_resource
def load_knowledge_base():
    """
    加载文档、切分并创建向量数据库。
    使用 cache_resource 确保只在第一次运行时执行。
    """
    with st.spinner("🧠 正在初始化知识库，请稍候..."):
        # 1. 获取当前脚本所在的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ⚠️ 请确保这个文件名和你上传到 GitHub 的文件名完全一致！
        file_name = "resume.txt" 
        file_path = os.path.join(current_dir, file_name)

        # 2. 检查文件是否存在
        if not os.path.exists(file_path):
            st.error(f"❌ 找不到文件: {file_name}。请检查文件名是否正确，或是否已上传到 GitHub。")
            return None

        # 3. 加载文档
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        except Exception as e:
            st.error(f"❌ 读取文件时出错: {e}")
            return None

        if not documents:
            st.error("❌ 文件内容为空，无法构建知识库！")
            return None

        # 4. 文本切分
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 5. 创建 Embeddings 对象 (使用 HuggingFace 的中文嵌入模型)
        # 这里使用一个轻量级的中文模型，适合本地和云端运行
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # 6. 创建向量数据库 (Chroma)
        # 注意：这里返回的是向量数据库对象，它拥有 as_retriever() 方法
        db = Chroma.from_documents(texts, embeddings)
        
        # 将 Embeddings 和 DB 一起返回，或者只返回 DB (Chroma 内部已包含 embeddings 信息)
        return db

# --- 5. 初始化大模型 ---
# 注意：这里不加缓存，或者根据需要缓存，但不要和 DB 混在一起
def get_llm(api_key):
    return ChatOpenAI(
        model_name="deepseek-chat",
        temperature=0.5,
        openai_api_key=api_key, # 显式传入 key
        openai_api_base="https://api.deepseek.com" # 显式传入 base url
    )

# --- 6. 应用主逻辑 ---
# 只有当用户输入了 Key 之后，才开始加载模型和数据库
if api_key and 'qa_chain' not in st.session_state:
    # 加载知识库
    vector_db = load_knowledge_base()
    
    if vector_db is not None:
        # 创建 LLM 实例
        llm = get_llm(api_key)
        
        # 创建检索问答链
        st.session先行.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3})
        )
        st.success("知识库加载完成，可以开始提问了！")
    else:
        st.error("知识库加载失败，请检查文件。")

# --- 7. 聊天界面 ---
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
                with st.spinner("正在思考..."):
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                st.write(response['result'])
            except Exception as e:
                st.error(f"出错了：{e}")
                st.info("提示：请检查 API Key 是否正确，或网络连接是否正常。")
