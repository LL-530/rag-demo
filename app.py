# -*- coding: utf-8 -*-
# -*- coding: gbk -*-
"""
RAG Demo - 基于 Flask + LangChain + ChromaDB 的检索增强生成系统
=====================================================================
功能：  1. 文档上传（txt / markdown / **PDF**）
       2. 文本分片 → Embedding → 存入 ChromaDB 向量库
       3. 用户提问 → 检索相关片段 → 调用 LLM 生成回答
       4. Web 应答界面 + 历史会话记录
       5. 流式输出（打字机效果）
       6. 引用来源显示（可点开查看原文片段）
       7. 多文档管理（列表查看 / 删除）
"""

import os
import json
import io
import datetime
import traceback
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, session, Response

# ---- LangChain & ChromaDB ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---- PDF 解析依赖 ----
import pypdf

# ---- 加载环境变量 ----
load_dotenv()  # 读取 .env 文件

# ---- 配置 ----
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 加载 config.json（用户通过前端设置面板保存的配置）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
APP_CONFIG = _load_config()

# 如果没有配置 API_KEY，启动时打印提醒但不阻断，以空接口查看界面
if not API_KEY:
    print("[提示] 未检测到 API_KEY，请在 .env 文件中配置后再启动。")

# ---- Flask 初始化 ----
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = DATA_DIR
# 上传大小限制：10MB（PDF 文件可能较大）
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ================================================================
# 核心：获取 LLM 实例（启动失败时捕获，避免阻断启动）
# ================================================================
def get_llm():
    """返回 ChatOpenAI 实例，如果未配置 API_KEY 则返回 None"""
    if not API_KEY:
        return None
    model_name = APP_CONFIG.get("model_name", "gpt-3.5-turbo")
    temperature = APP_CONFIG.get("temperature", 0.3)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
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
        model="step-1-8k",
    )


# ================================================================
# 核心：加载向量库
# ================================================================
def get_vectorstore():
    """
    从持久化目录加载 ChromaDB 向量库。
    如果数据目录不存在或没有可用 embeddings，返回 None。
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
# 辅助：文本分片
# ================================================================
def split_text(text: str) -> list[Document]:
    """
    将长文本按段落切分为小块，用于向量化和检索。
    分片参数：
      chunk_size      = 500 字符
      chunk_overlap   = 50  字符（保留上下文衔接）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "？", "！", " ", ""],
        length_function=len,
    )
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    return docs


# ================================================================
# 辅助：切分文本并附加来源文件名元数据
# ================================================================
def make_docs_with_source(raw_text: str, filename: str,
                          page_map: dict = None) -> list[Document]:
    """
    切分文本，并为每个 chunk 附加 source 元数据（用于引用来源显示）。
    可选地传入 page_map，为每个 chunk 附加 page_number 元数据（PDF 用）。

    参数：
      raw_text  : 原始文本
      filename  : 来源文件名
      page_map  : {chunk_index: page_number} 的映射字典，可选
    返回：
      list[Document] 每个 chunk 都带有 source 元数据
    """
    docs = split_text(raw_text)
    for i, d in enumerate(docs):
        d.metadata["source"] = filename
        # 如果提供了 page_map，附加对应页码
        if page_map is not None and i in page_map:
            d.metadata["page_number"] = page_map[i]
    return docs



# ================================================================
# 新增：多轮对话上下文构建
# ================================================================
def build_chat_context(sess, max_chars: int = 1000) -> str:
    """
    从 Flask session 中提取最近 5 轮对话（user + assistant 各算 1 轮，
    即最多 10 条消息），格式化为 prompt 前缀。

    设计说明：
      - 为什么要限制 max_chars（默认 1000）？
        prompt 越长，消耗的 token 越多，响应越慢、费用越高。
        限制在约 1000 字符（~250-400 token）可以在保证上下文连贯的同时
        控制成本，避免长对话拖慢响应。

    返回：
      格式化后的上下文字符串；若无历史对话则返回空字符串。
    """
    history = sess.get("chat_history", [])
    if not history:
        return ""

    # 取最近 10 条（5 轮 user + assistant）
    recent = history[-10:]

    # 格式化为纯文本：用户：xxx / 助手：xxx
    lines: list[str] = []
    for msg in recent:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}：{msg['content']}")

    context_text = "\n".join(lines)

    # 字符数超限时，从最早的对话开始裁剪
    if len(context_text) > max_chars:
        while len(context_text) > max_chars and len(lines) > 1:
            lines.pop(0)
            context_text = "\n".join(lines)

    if not lines:
        return ""

    return (
        "以下是之前的对话记录：\n"
        + context_text
        + "\n\n"
        + "现在回答用户的新问题："
    )


# ================================================================
# 新增：从 PDF 文件中提取文本
# ================================================================
def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, dict]:
    """
    使用 pypdf 解析 PDF 文件，提取全部文本内容。

    同时构建 chunk_index → page_number 的映射表，
    用于在引用来源中显示 PDF 页码。

    参数：
      file_bytes  : PDF 文件的原始二进制内容
    返回：
      (text, page_map)
        text     : 提取的完整文本（字符串）
        page_map : {chunk_index: page_number}，若出错则为空字典
    异常：
      若 PDF 加密或无法解析，抛出 ValueError 并附带友好提示
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"无法解析 PDF 文件：{e}")

    # 检查是否加密
    if reader.is_encrypted:
        raise ValueError("该 PDF 文件已加密，请提供未加密的文件后重试。")

    # 逐页提取文本，同时记录每页的字符范围
    page_texts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        page_texts.append(t)

    full_text = "\n\n".join(page_texts)

    # 构建 chunk_index → page_number 映射
    # 策略：遍历全文，每当遇到某页文本的起始位置时，记录该 chunk 属于哪一页
    page_map: dict[int, int] = {}
    char_pos = 0          # 当前在整个 full_text 中的字符偏移量
    chunk_size = 500      # 与 split_text 保持一致
    chunk_overlap = 50

    for page_idx, page_t in enumerate(page_texts):
        page_start = full_text.find(page_t, char_pos)
        if page_start == -1:
            # 极端情况：重复文本导致找不到精确位置，跳过该页
            char_pos += len(page_t) + 2
            continue

        # 该页对应的 chunk 起始索引
        # chunk_start_char 是该 chunk 在整个文本中的字符位置
        # chunk_index = chunk_start_char // (chunk_size - chunk_overlap)
        first_chunk_idx = page_start // (chunk_size - chunk_overlap)
        # 该页最后一个字符对应的 chunk
        page_end = page_start + len(page_t)
        last_chunk_idx = page_end // (chunk_size - chunk_overlap)

        for cidx in range(first_chunk_idx, last_chunk_idx + 1):
            page_map[cidx] = page_idx + 1  # 页码从 1 开始

        char_pos = page_end + 2  # +2 是因为 join 时插入了 "\n\n"

    return full_text, page_map


# ================================================================
# 辅助：初始化示例数据
# ================================================================
def init_sample_data():
    """
    项目首次运行时，将 data/example_doc.txt 写入向量库，
    确保打开页面后可直接提问，无需手动上传。
    """
    embeddings = get_embeddings()
    if embeddings is None:
        print("[初始化数据] 未配置 API_KEY，跳过示例数据初始化。")
        return
    example_path = os.path.join(DATA_DIR, "example_doc.txt")
    if not os.path.exists(example_path):
        print("[初始化数据] 未找到 example_doc.txt，跳过。")
        return
    vs = get_vectorstore()
    if vs is not None:
        print("[初始化数据] 向量库已存在，跳过初始化。")
        return
    print("[初始化数据] 正在初始化示例数据...")
    with open(example_path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = make_docs_with_source(text, "example_doc.txt")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="rag_documents",
    )
    print(f"[初始化数据] 已将 {len(docs)} 个文本块写入向量库。")


# ================================================================
# 路由：设置管理
# ================================================================
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        cfg = _load_config()
        api_key = cfg.get("api_key", "")
        masked = (api_key[:4] + "****" + api_key[-4:]) if len(api_key) > 8 else ""
        return jsonify({"status": "ok", "settings": {**cfg, "api_key": masked}})
    data = request.get_json(force=True) or {}
    cfg = _load_config()
    if "api_key" in data and data["api_key"] and not data["api_key"].startswith("****"):
        cfg["api_key"] = data["api_key"]
    for k in ["base_url", "model_name", "temperature"]:
        if k in data:
            cfg[k] = data[k]
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})

# ================================================================
# 路由：主页
# ================================================================
@app.route("/")
def index():
    """渲染问答主页面"""
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html")


# ================================================================
# 路由：上传文档（支持 .txt / .md / **.pdf**）
# ================================================================
@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    接收用户上传的 txt / markdown / PDF 文件，切分后存入向量库。
    JSON 响应：{"status": "ok", "chunks": 12, "message": "..."}
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return jsonify({"status": "err", "message": "未配置 API_KEY，请先在 .env 文件中配置。"}), 500

    if "file" not in request.files:
        return jsonify({"status": "err", "message": "请先选择要上传的文件。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "err", "message": "文件名为空。"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    # ---- 支持的文件格式 ----
    if ext not in [".txt", ".md", ".pdf"]:
        return jsonify({"status": "err", "message": "仅支持 .txt、.md 或 .pdf 文件。"}), 400

    try:
        raw_bytes = file.read()
        page_map: dict[int, int] = {}   # chunk_index → page_number（PDF 用）

        # ---- 根据文件类型解析文本 ----
        if ext == ".pdf":
            text, page_map = extract_text_from_pdf(raw_bytes)
        else:
            # txt / md：以 UTF-8 解码，不可见字符用替代符
            text = raw_bytes.decode("utf-8", errors="replace")

        # ---- 保存原始文件到 data 目录（备份用）----
        os.makedirs(DATA_DIR, exist_ok=True)
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        backup_path = os.path.join(DATA_DIR, safe_filename)
        with open(backup_path, "wb") as bf:
            bf.write(raw_bytes)

        # ---- 切分并附加元数据 ----
        docs = make_docs_with_source(text, filename, page_map=page_map)
        if not docs:
            return jsonify({"status": "err", "message": "文件内容为空或无法解析。"}), 400

        # ---- 写入向量库 ----
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

    except ValueError as ve:
        # 已知业务错误（如加密 PDF），直接返回友好提示
        return jsonify({"status": "err", "message": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "err", "message": f"处理失败：{str(e)}"}), 500


# ================================================================
# 路由：粘贴文本
# ================================================================
@app.route("/api/paste", methods=["POST"])
def paste_text():
    """
    接收用户粘贴的文本，分片后存入向量库。
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
            "message": f"文本已成功切分为 {len(docs)} 个片段并存入向量库。",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "err", "message": f"处理失败：{str(e)}"}), 500


# ================================================================
# 路由：流式问答（Server-Sent Events）
# ================================================================
@app.route("/api/chat-stream", methods=["POST"])
def chat_stream():
    """
    流式问答接口，使用 Server-Sent Events (SSE) 逐字返回回答。
    支持无向量库降级：当检索失败时，直接以纯对话模式回答。
    SSE 数据格式：
      data: {"type": "token",    "content": "字"}
      data: {"type": "sources",  "sources": [{"file": "...", "idx": 0, "text": "...", "page": null}]}
      data: {"type": "done"}
      data: {"type": "error",    "message": "..."}
    """
    llm = get_llm()
    vectorstore = get_vectorstore()

    if llm is None:
        def _err():
            yield f"data: {json.dumps({'type': 'error', 'message': '模型未就绪，请检查配置。'})}\n\n"
        return Response(_err(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        def _err2():
            yield f"data: {json.dumps({'type': 'error', 'message': '问题不能为空。'})}\n\n"
        return Response(_err2(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- 检索相关片段（失败则降级为纯对话）----
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in relevant_docs)
    except Exception:
        relevant_docs = []
        context = ""

    # ---- 构造 Prompt（含多轮对话上下文）----
    # 从 session 中获取历史对话，嵌入到 prompt 前缀中
    chat_ctx = build_chat_context(session)

    if context:
        if chat_ctx:
            # 有历史对话：上下文 + 参考资料 + 用户新问题
            prompt = f"""{chat_ctx}
---参考资料：---
{context}
---
请基于以上对话上下文和参考资料回答用户的新问题。
如果参考资料不足以回答，请如实说明，不要编造信息。
请用中文回答，结果简洁，条理清晰。"""
        else:
            # 无历史对话：保持原有 prompt 不变
            prompt = f"""你是一位专业、友善的 AI 答疑助手。请基于以下参考资料回复用户问题。
如果参考资料不足以回答某个问题，请如实说明"根据现有资料无法回答该问题"，不要编造信息。
---
参考资料：
{context}
---

用户问题：{question}

请用中文回答，结果简洁，条理清晰。"""
    else:
        # 无检索结果：纯对话模式
        if chat_ctx:
            prompt = f"{chat_ctx}\n\n用户问题：{question}\n\n请基于对话上下文回答，不要编造信息。请用中文回答，结果简洁，条理清晰。"
        else:
            prompt = f"你是一位专业、友善的 AI 答疑助手。\n\n用户问题：{question}\n\n请用中文回答，结果简洁，条理清晰。"

        # 无来源，直接流式返回
        def _no_rag():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            for chunk in llm.stream(prompt):
                token = chunk.content or ""
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        return Response(_no_rag(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- 收集来源信息（含可选页码）----
    sources = []
    for idx, doc in enumerate(relevant_docs):
        label = doc.metadata.get("source", "未知文档")
        snippet = doc.page_content.strip()
        page_num = doc.metadata.get("page_number")  # PDF 才有，None 表示非 PDF
        sources.append({
            "file": label,
            "idx": idx + 1,
            "text": snippet,
            "page": page_num,
        })

    # ---- 流式生成 ----
    def generate():
        # 先发送来源信息
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        # 逐 token 流式返回回答
        for chunk in llm.stream(prompt):
            token = chunk.content or ""
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    用户提问 → 检索相关文档 → 构造 prompt → 调用 LLM 生成回答
    支持无向量库降级：当检索失败时，直接以纯对话模式回答。
    JSON 请求体：{"question": "..."}
    响应：{"status": "ok", "answer": "...", "sources": [...]}
    """
    llm = get_llm()
    vectorstore = get_vectorstore()

    if llm is None:
        return jsonify({
            "status": "err",
            "answer": "模型未就绪，请检查配置。",
        }), 500

    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"status": "err", "answer": "问题不能为空。"}), 400

    # ---- 检索相关文档（失败则降级为纯对话）----
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in relevant_docs)
    except Exception:
        relevant_docs = []
        context = ""

    # ---- 构造 Prompt（含多轮对话上下文）----
    chat_ctx = build_chat_context(session)

    if context:
        if chat_ctx:
            prompt = f"""{chat_ctx}
---参考资料：---
{context}
---
请基于以上对话上下文和参考资料回答用户的新问题。
如果参考资料不足以回答，请如实说明，不要编造信息。
请用中文回答，结果简洁，条理清晰。"""
        else:
            prompt = f"""你是一位专业、友善的 AI 答疑助手。请基于以下参考资料回复用户问题。
如果参考资料不足以回答某个问题，请如实说明"根据现有资料无法回答该问题"，不要编造信息。
---
参考资料：
{context}
---

用户问题：{question}

请用中文回答，结果简洁，条理清晰。"""
    else:
        # 无检索结果：纯对话模式
        if chat_ctx:
            prompt = f"{chat_ctx}\n\n用户问题：{question}\n\n请基于对话上下文回答，不要编造信息。请用中文回答，结果简洁，条理清晰。"
        else:
            prompt = f"你是一位专业、友善的 AI 答疑助手。\n\n用户问题：{question}\n\n请用中文回答，结果简洁，条理清晰。"

        # 无来源，直接返回
        answer = llm.invoke(prompt).content.strip()
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
        return jsonify({"status": "ok", "answer": answer, "sources": []})

    # ---- 调用 LLM ----
    response = llm.invoke(prompt)
    answer = response.content.strip()

    # ---- 收集来源（含可选页码）----
    sources = []
    for idx, doc in enumerate(relevant_docs):
        label = doc.metadata.get("source", "未知文档")
        snippet = doc.page_content.strip()
        page_num = doc.metadata.get("page_number")
        sources.append({
            "file": label,
            "idx": idx + 1,
            "text": snippet,
            "page": page_num,
        })

    # ---- 存入会话历史 ----
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



# ================================================================
# 文档管理 - 获取文档列表
# ================================================================
@app.route("/api/documents", methods=["GET"])
def list_documents():
    """
    返回向量库中所有文档的统计信息（按来源文件名去重）。
    """
    try:
        vs = get_vectorstore()
        if vs is None:
            return jsonify({"status": "ok", "documents": []})

        # 直接访问底层 collection 获取重复数据
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
# 文档管理 - 删除指定文档
# ================================================================
@app.route("/api/documents", methods=["DELETE"])
def delete_document():
    """
    从向量库中删除某文档的所有片段。
    JSON 请求体：{"filename": "要删除的文件名"}
    example_doc.txt 内置示例不可删除。
    """
    PROTECTED = {"example_doc.txt", "data/example_doc.txt"}

    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"status": "err", "message": "请指定要删除的文件名。"}), 400
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
# 路由：获取 / 追加会话历史
# ================================================================
@app.route("/api/history", methods=["GET", "POST"])
def history_handler():
    """
    GET  : 返回当前 session 的对话历史
    POST : 将新的 user/assistant 对话追加到 session（供流式接口前端调用）
    """
    if request.method == "GET":
        return jsonify({"history": session.get("chat_history", [])})

    # POST：追加对话
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    answer   = (data.get("answer")   or "").strip()
    if not question or not answer:
        return jsonify({"status": "err", "message": "缺少 question 或 answer 字段。"}), 400

    chat_history = session.get("chat_history", [])
    now = __import__("datetime").datetime.now().strftime("%H:%M")
    chat_history.append({"role": "user",      "content": question, "time": now})
    chat_history.append({"role": "assistant", "content": answer,   "time": now})
    session["chat_history"] = chat_history[-20:]
    return jsonify({"status": "ok"})


# ================================================================
# 路由：清空会话历史
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

    # 初始化示例数据（首次且 API_KEY 可用时）
    init_sample_data()

    print("\n" + "=" * 50)
    print("  RAG Demo 启动中...")
    print(f"  访问地址：http://127.0.0.1:5000")
    print(f"  数据目录：{DATA_DIR}")
    print(f"  向量库目录：{PERSIST_DIR}")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
