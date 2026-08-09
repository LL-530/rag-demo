#!/usr/bin/env python3
"""Replace chat() in rag-demo/app.py with no-RAG fallback version."""

path = 'app.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find chat() function boundaries
start_marker = 'def chat():'
# Find the next top-level definition or route after chat()
# Look for '\n\n@app.route' or '\n\ndef ' after the function
start_idx = content.find(start_marker)
if start_idx == -1:
    raise SystemExit('chat() not found')

# Find end of chat() - next blank line followed by @app.route or def at top level
end_idx = content.find('\n\n# ====', start_idx)
if end_idx == -1:
    end_idx = content.find('\n\n@app.route', start_idx)
if end_idx == -1:
    end_idx = content.find('\n\ndef ', start_idx)
if end_idx == -1:
    raise SystemExit('Cannot find end of chat()')

new_func = '''def chat():
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
        context = "\\n\\n".join(d.page_content for d in relevant_docs)
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
            prompt = f"{chat_ctx}\\n\\n用户问题：{question}\\n\\n请基于对话上下文回答，不要编造信息。请用中文回答，结果简洁，条理清晰。"
        else:
            prompt = f"你是一位专业、友善的 AI 答疑助手。\\n\\n用户问题：{question}\\n\\n请用中文回答，结果简洁，条理清晰。"

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

'''

new_content = content[:start_idx] + new_func + content[end_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Replaced chat()')
