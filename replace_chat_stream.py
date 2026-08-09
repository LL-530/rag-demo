#!/usr/bin/env python3
"""Replace chat_stream() in rag-demo/app.py with no-RAG fallback version."""

path = 'app.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find chat_stream function boundaries
start_marker = 'def chat_stream():'
end_marker = '\n@app.route("/api/chat", methods=["POST"])'

start_idx = content.find(start_marker)
if start_idx == -1:
    raise SystemExit('chat_stream() not found')

end_idx = content.find(end_marker, start_idx)
if end_idx == -1:
    raise SystemExit('chat() route not found after chat_stream')

# New chat_stream implementation
new_func = '''def chat_stream():
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
            yield f"data: {json.dumps({'type': 'error', 'message': '模型未就绪，请检查配置。'})}\\n\\n"
        return Response(_err(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        def _err2():
            yield f"data: {json.dumps({'type': 'error', 'message': '问题不能为空。'})}\\n\\n"
        return Response(_err2(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- 检索相关片段（失败则降级为纯对话）----
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        context = "\\n\\n".join(d.page_content for d in relevant_docs)
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
            prompt = f"{chat_ctx}\\n\\n用户问题：{question}\\n\\n请基于对话上下文回答，不要编造信息。请用中文回答，结果简洁，条理清晰。"
        else:
            prompt = f"你是一位专业、友善的 AI 答疑助手。\\n\\n用户问题：{question}\\n\\n请用中文回答，结果简洁，条理清晰。"

        # 无来源，直接流式返回
        def _no_rag():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\\n\\n"
            for chunk in llm.stream(prompt):
                token = chunk.content or ""
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\\n\\n"
            yield "data: {\\"type\\": \\"done\\"}\\n\\n"
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
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\\n\\n"
        # 逐 token 流式返回回答
        for chunk in llm.stream(prompt):
            token = chunk.content or ""
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\\n\\n"
        yield "data: {\\"type\\": \\"done\\"}\\n\\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

'''

# Replace the entire chat_stream function
new_content = content[:start_idx] + new_func + content[end_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Replaced chat_stream()')
