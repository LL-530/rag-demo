"""
RAG Demo - 基于 Flask + LangChain + ChromaDB 的检索增强生成问答系统
=====================================================================
功能：
  1. 文档上传（txt / markdown）或粘贴文本
  2. 文本切分 → Embedding → 存入 ChromaDB 向量库
  3. 用户提问 → 检索相关片段 → 调用 LLM 生成回答
  4. Web 问答界面 + 历史对话记录
  5. ★ 流式输出（打字机效果）
  6. ★ 引用来源展示（可展开查看原文片段）
  7. ★ 多文档管理（列表 / 删除）
"""

import os
import json
import datetime
import traceback
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, session, Response

# ---- LangChain & ChromaDB ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---- 加载环境变量 ----
load_dotenv()  # 读取 .env 文件

# ---- 配置 ----
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 如果没配 API_KEY，启动时打印警告但不阻断，以便空跑查看界面
if not API_KEY:
    print("[警告] 未检测到 API_KEY，请在 .env 文件中配置后再启动。")

# ---- Flask 初始化 ----
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = DATA_DIR
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ================================================================
# 核心：获取 LLM 实例（延迟初始化，避免启动时就报错）
# ================================================================
def get_llm():
    """返回 ChatOpenAI 实例，如果 API_KEY 未配置则返回 None"""
    if not API_KEY:
        return None
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        api_key=API_KEY,
        base_url=BASE_URL,
    )


# ================================================================
# 核心：获取 Embedding 实例
# ================================================================
def get_embeddings():
    """返回 OpenAIEmbeddings 实例，未配置时返回 None"""
    if not API_KEY:
        return None
    return OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


# ================================================================
# 核心：加载向量库
# ================================================================
def get_vectorstore():
    """
    加载持久化 ChromaDB 向量库。
    如果数据库不存在且没有可用 embeddings，返回 None。
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return None
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        return None
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="rag_documents",
    )


# ================================================================
# 辅助：文本切分
# ================================================================
def split_text(text: str) -> list[Document]:
    """
    将长文本按段落切分为小块，便于向量化和检索。
    切分参数：
      chunk_size      = 500 字符
      chunk_overlap   = 50  字符（保留上下文衔接）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""],
        length_function=len,
    )
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    return docs


# ================================================================
# 辅助：切分文本并附带来源文件名元数据
# ================================================================
def make_docs_with_source(raw_text: str, filename: str) -> list[Document]:
    """切分文本，并为每个 chunk 附加 source 元数据（用于引用来源展示）。"""
    docs = split_text(raw_text)
    for d in docs:
        d.metadata["source"] = filename
    return docs


# ================================================================
# 辅助：初始化示例数据
# ================================================================
def init_sample_data():
    """
    项目首次运行时，将 data/example_doc.txt 写入向量库，
    确保打开网页后立刻可以体验问答，无需手动上传。
    """
    embeddings = get_embeddings()
    if embeddings is None:
        print("[示例数据] 未配置 API_KEY，跳过示例数据初始化。")
        return
    example_path = os.path.join(DATA_DIR, "example_doc.txt")
    if not os.path.exists(example_path):
        print("[示例数据] 未找到 example_doc.txt，跳过。")
        return
    vs = get_vectorstore()
    if vs is not None:
        print("[示例数据] 向量库已存在，跳过初始化。")
        return
    print("[示例数据] 正在初始化示例数据...")
    with open(example_path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = make_docs_with_source(text, "example_doc.txt")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="rag_documents",
    )
    print(f"[示例数据] 已将 {len(docs)} 个文本块写入向量库。")


# ================================================================
# 路由：主页
# ================================================================
@app.route("/")
def index():
    """渲染问答页面"""
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html")


# ================================================================
# 路由：上传文档
# ================================================================
@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    接收用户上传的 txt/markdown 文件，切分后存入向量库。
    JSON 响应：{"status": "ok", "chunks": 12, "message": "..."}
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return jsonify({"status": "err", "message": "未配置 API_KEY，请先在 .env 文件中配置。"}), 500

    if "file" not in request.files:
        return jsonify({"status": "err", "message": "请选择要上传的文件。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "err", "message": "文件名为空。"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".txt", ".md"]:
        return jsonify({"status": "err", "message": "仅支持 .txt 或 .md 文件。"}), 400

    try:
        raw_bytes = file.read()
        text = raw_bytes.decode("utf-8", errors="replace")
        docs = make_docs_with_source(text, filename)
        if not docs:
            return jsonify({"status": "err", "message": "文件内容为空或无法解析。"}), 400

        vectorstore = get_vectorstore()
        if vectorstore is None:
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=PERSIST_DIR,
                collection_name="rag_documents",
            )
        else:
            vectorstore.add_documents(docs)
            vectorstore.persist()

        return jsonify({
            "status": "ok",
            "chunks": len(docs),
            "message": f"「{filename}」已成功解析为 {len(docs)} 个片段并存入向量库。",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "err", "message": f"处理失败：{str(e)}"}), 500


# ================================================================
# 路由：粘贴文本
# ================================================================
@app.route("/api/paste", methods=["POST"])
def paste_text():
    """
    接收用户粘贴的文本，切分后存入向量库。
    JSON 请求体：{"text": "..."}
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return jsonify({"status": "err", "message": "未配置 API_KEY，请先在 .env 文件中配置。"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"status": "err", "message": "请求体缺少 text 字段。"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"status": "err", "message": "文本内容为空。"}), 400

    try:
        docs = make_docs_with_source(text, "用户粘贴文本")
        if not docs:
            return jsonify({"status": "err", "message": "文本为空，无法处理。"}), 400

        vectorstore = get_vectorstore()
        if vectorstore is None:
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=PERSIST_DIR,
                collection_name="rag_documents",
            )
        else:
            vectorstore.add_documents(docs)
            vectorstore.persist()

        return jsonify({
            "status": "ok",
            "chunks": len(docs),
            "message": f"文本已成功解析为 {len(docs)} 个片段并存入向量库。",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "err", "message": f"处理失败：{str(e)}"}), 500


# ================================================================
# ★ 流式问答（Server-Sent Events）
# ================================================================
@app.route("/api/chat-stream", methods=["POST"])
def chat_stream():
    """
    流式问答接口，使用 Server-Sent Events (SSE) 逐字返回回答。

    SSE 数据格式：
      data: {"type": "token",    "content": "字"}
      data: {"type": "sources",  "sources": [{"file": "...", "idx": 0, "text": "..."}]}
      data: {"type": "done"}
      data: {"type": "error",    "message": "..."}
    """
    llm = get_llm()
    vectorstore = get_vectorstore()

    if llm is None or vectorstore is None:
        def _err():
            yield f"data: {json.dumps({'type': 'error', 'message': '模型或向量库未就绪，请检查配置。'})}\n\n"
        return Response(_err(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        def _err2():
            yield f"data: {json.dumps({'type': 'error', 'message': '问题不能为空。'})}\n\n"
        return Response(_err2(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        # ---- 检索相关片段 ----
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in relevant_docs)

        # ---- 构造 Prompt ----
        prompt = f"""你是一个专业、友好的 AI 问答助手。请基于以下参考资料回答用户问题。
如果参考资料不足以回答问题，请明确说明"根据现有资料无法回答该问题"，
不要编造信息。

--- 参考资料 ---
{context}
--- 参考资料结束 ---

用户问题：{question}

请用中文回答，结构清晰，条理分明："""

        # ---- 收集来源信息 ----
        sources = []
        for idx, doc in enumerate(relevant_docs):
            label = doc.metadata.get("source", "未知文档")
            snippet = doc.page_content.strip()
            sources.append({"file": label, "idx": idx + 1, "text": snippet})

        # ---- 流式生成 ----
        def generate():
            # 先推送来源信息
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            # 逐 token 流式推送回答
            for chunk in llm.stream(prompt):
                token = chunk.content or ""
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    except Exception as e:
        traceback.print_exc()
        def _err3():
            yield f"data: {json.dumps({'type': 'error', 'message': f'生成回答时出错：{str(e)}'})}\n\n"
        return Response(_err3(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ================================================================
# 路由：问答（非流式，保持兼容）
# ================================================================
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    用户提问 → 检索向量库 → 构造 prompt → 调用 LLM 生成回答
    JSON 请求体：{"question": "..."}
    响应：{"status": "ok", "answer": "...", "sources": [...]}
    """
    llm = get_llm()
    vectorstore = get_vectorstore()

    if llm is None or vectorstore is None:
        return jsonify({
            "status": "err",
            "answer": "模型或向量库未就绪。请确保：\n① 已配置 .env 文件中的 API_KEY\n② 已上传文档或内置示例数据已初始化",
        }), 500

    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"status": "err", "answer": "问题不能为空。"}), 400

    try:
        # ---- 检索相关文档片段（top-k = 3）----
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in relevant_docs)

        # ---- 构造 Prompt（中文提示词）----
        prompt = f"""你是一个专业、友好的 AI 问答助手。请基于以下参考资料回答用户问题。
如果参考资料不足以回答问题，请明确说明"根据现有资料无法回答该问题"，
不要编造信息。

--- 参考资料 ---
{context}
--- 参考资料结束 ---

用户问题：{question}

请用中文回答，结构清晰，条理分明："""

        # ---- 调用 LLM ----
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # ---- 收集来源 ----
        sources = []
        for idx, doc in enumerate(relevant_docs):
            label = doc.metadata.get("source", "未知文档")
            snippet = doc.page_content.strip()
            sources.append({"file": label, "idx": idx + 1, "text": snippet})

        # ---- 存入对话历史 ----
        chat_history = session.get("chat_history", [])
        chat_history.append({
            "role": "user",
            "content": question,
            "time": datetime.datetime.now().strftime("%H:%M"),
        })
        chat_history.append({
            "role": "assistant",
            "content": answer,
            "time": datetime.datetime.now().strftime("%H:%M"),
        })
        session["chat_history"] = chat_history[-20:]

        return jsonify({"status": "ok", "answer": answer, "sources": sources})

    except Exception e:
        traceback.print_exc()
        return jsonify({"status": "err", "answer": f"生成回答时出错：{str(e)}"}), 500


# ================================================================
# ★ 文档管理 - 获取文档列表
# ================================================================
@app.route("/api/documents", methods=["GET"])
def list_documents():
    """
    返回向量库中所有文档的统计信息（按来源文件名去重聚合）。
    """
    try:
        vs = get_vectorstore()
        if vs is None:
            return jsonify({"status": "ok", "documents": []})

        # 直接访问底层 collection 获取去重元数据
        collection = vs._collection
        result = collection.get(include=["metadatas"])

        doc_stats = {}  # filename -> chunk count
        for meta in (result.get("metadatas") or []):
            if isinstance(meta, dict):
                src = meta.get("source", "未知文档")
                doc_stats[src] = doc_stats.get(src, 0) + 1

        documents = [
            {"filename": name, "chunks": count}
            for name, count in sorted(doc_stats.items())
        ]
        return jsonify({"status": "ok", "documents": documents})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "err", "message": str(e)}), 500


# ================================================================
# ★ 文档管理 - 删除指定文档
# ================================================================
@app.route("/api/documents", methods=["DELETE"])
def delete_document():
    """
    从向量库中删除某个文档的所有片段。
    JSON 请求体：{"filename": "要删除的文件名"}
    example_doc.txt 受保护，不允许删除。
    """
    PROTECTED = {"example_doc.txt", "data/example_doc.txt"}

    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "err", "message": "请提供要删除的文件名。"}), 400
    if filename in PROTECTED:
        return jsonify({"status": "err", "message": "内置示例文档不可删除。"}), 403

    try:
        vs = get_vectorstore()
        if vs is None:
            return jsonify({"status": "err", "message": "向量库为空，无法删除。"}), 400

        collection = vs._collection
        result = collection.get(where={"source": filename}, include=[])
        ids_to_delete = result.get("ids", [])

        if not ids_to_delete:
            return jsonify({"status": "ok", "message": f"「{filename}」未找到或已被删除。"})

        collection.delete(ids=ids_to_delete)
        vs.persist()

        return jsonify({
            "status": "ok",
            "message": f"「{filename}」已删除，共移除 {len(ids_to_delete)} 个片段。",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "err", "message": f"删除失败：{str(e)}"}), 500


# ================================================================
# 路由：获取对话历史
# ================================================================
@app.route("/api/history", methods=["GET"])
def get_history():
    """返回当前 session 的对话历史"""
    return jsonify({"history": session.get("chat_history", [])})


# ================================================================
# 路由：清空对话历史
# ================================================================
@app.route("/api/clear", methods=["POST"])
def clear_history():
    """清空当前 session 的对话历史"""
    session["chat_history"] = []
    return jsonify({"status": "ok", "message": "对话历史已清空。"})


# ================================================================
# 路由：获取向量库状态
# ================================================================
@app.route("/api/status", methods=["GET"])
def get_status():
    """返回向量库和模型配置状态"""
    embeddings_ready = get_embeddings() is not None
    llm_ready = get_llm() is not None
    vs = get_vectorstore()
    has_data = vs is not None
    return jsonify({
        "embeddings_ready": embeddings_ready,
        "llm_ready": llm_ready,
        "vectorstore_ready": has_data,
        "api_configured": bool(API_KEY),
        "base_url": BASE_URL,
    })


# ================================================================
# 启动
# ================================================================
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # 初始化示例数据（仅首次、且 API_KEY 可用时）
    init_sample_data()

    print("\n" + "=" * 50)
    print("  RAG Demo 启动中...")
    print(f"  访问地址：http://127.0.0.1:5000")
    print(f"  数据目录：{DATA_DIR}")
    print(f"  向量库目录：{PERSIST_DIR}")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
