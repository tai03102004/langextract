import os
import gradio as gr
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# Lấy API key và base URL
API_KEY = os.getenv("RAG_QWEN3_NEXT_80B_A3B_THINKING")
BASE_URL = os.getenv("LLM_BASE_URL")
# Tên model generation (có thể thay bằng qwen-max, qwen-plus, ...)
LLM_MODEL = "qwen3-next-80b-a3b-thinking"
# Tên embedding model của DashScope (hỗ trợ tương thích OpenAI)
EMBEDDING_MODEL = "text-embedding-v3"  # hoặc "text-embedding-v2"

# 1. Xây dựng retriever từ tài liệu
def build_retriever(docs_path="documents/"):
    if os.path.isfile(docs_path):
        loader = TextLoader(docs_path)
    else:
        loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    # Embedding dùng OpenAIEmbeddings nhưng trỏ base URL và model riêng
    embeddings = OpenAIEmbeddings(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=EMBEDDING_MODEL,
        chunk_size=1  # tránh lỗi chunk size
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. LLM và QA chain
llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    model_name=LLM_MODEL,
    temperature=0
)
retriever = build_retriever("documents/")  # thay bằng thư mục/file của bạn
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 3. Hàm chat
def format_context(docs):
    if not docs:
        return "*Không tìm thấy tài liệu liên quan.*"
    html = "<div style='background:#f9f9f9; padding:10px; border-radius:8px;'>"
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        html += f"<p><b>📄 {source}:</b><br>{doc.page_content[:300]}...</p><hr>"
    html += "</div>"
    return html

def answer_question(message, history):
    result = qa_chain.invoke({"query": message})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, format_context(sources)

# 4. Giao diện Gradio
with gr.Blocks(title="📚 RAG Qwen Assistant") as demo:
    gr.Markdown("# 🤖 RAG Assistant with Qwen3")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=500)
            msg = gr.Textbox(label="Câu hỏi", placeholder="Nhập câu hỏi...")
            clear = gr.Button("Xóa lịch sử")
        with gr.Column(scale=1):
            context_box = gr.HTML(label="Ngữ cảnh liên quan")

    def respond(message, chat_history):
        if not message:
            return "", chat_history, ""
        answer, context_html = answer_question(message, chat_history)
        chat_history.append((message, answer))
        return "", chat_history, context_html

    msg.submit(respond, [msg, chatbot], [msg, chatbot, context_box])
    clear.click(lambda: ([], ""), None, [chatbot, context_box])

demo.launch(inbrowser=True)