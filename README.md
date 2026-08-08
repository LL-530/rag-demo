# 📚 RAG Demo - 检索增强生成问答系统

> 企业级 / 个人学习 Demo  
> 技术栈：Flask + LangChain + ChromaDB + OpenAI 兼容 API

---

## 🎯 项目简介

RAG（Retrieval-Augmented Generation，检索增强生成）是目前最主流的 AI 应用架构之一。  
本项目实现了一个**完整的 RAG 问答 Demo**，涵盖：

| 步骤 | 说明 |
|------|------|
| 📤 文档上传 | 支持 `.txt` / `.md` 文件和自动分片入库 |
| ✂️ 文本分片 | 将长文本按段落切分为 500 字符的语义块 |
| 🔢 向量化 | 调用 Embedding 模型生成向量 |
| 🗄️ 向量存储 | 向量存入本地 ChromaDB（无需数据库） |
| 🔍 检索匹配 | 用户提问 → 检索 top-3 相关片段 |
| 🤖 LLM 生成 | 结合检索结果调用 LLM 生成结构化回答 |
| 💬 多轮对话 | 浏览器 Session 自动缓存最近 20 条记录 |
| ⚡ **流式输出** | **逐字显示回答，打字机效果** |
| 📌 **引用来源** | **回答末尾展示引用文件，点击展开看原文片段** |
| 📁 **文档管理** | **列出所有已上传文档，支持删除** |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境变量示例
copy .env.example .env
```

打开 `.env`，填入你的 API Key 和 Base URL：

```ini
API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BASE_URL=https://api.stepfun.com/v1
```

> **配置说明**
>
> - **使用 OpenAI 官方接口**：只需填写 `API_KEY`，`BASE_URL` 保持默认
> - **使用第三方中转服务**（DeepSeek / OneAPI / OpenRouter 等）：将 `BASE_URL` 改为对应服务的 API 地址，如 `https://api.deepseek.com/v1`

### 3. 启动服务

```bash
python app.py
```

看到如下日志即表示启动成功：

```
==================================================
  RAG Demo 启动中...
  访问地址：http://127.0.0.1:5000
  数据目录：C:\Users\xxx\rag-demo\data
  向量库目录：C:\Users\xxx\rag-demo\chroma_db
==================================================
```

### 4. 打开浏览器

访问 [http://127.0.0.1:5000](http://127.0.0.1:5000)

首次启动时，系统会自动将 `data/example_doc.txt`（人类AI发展简史）写入向量库，打开页面即可直接提问。

---

## ✨ 核心功能详解

### 流式输出（打字机效果）

采用 **Server-Sent Events (SSE)** 实现实时逐字返回，结合前端 `ReadableStream` 逐 token 接收并渲染：
- **后端**：新增 `/api/chat-stream` 接口，使用 `Response + generator` 流式返回
- **前端**：使用 `fetch + ReadableStream` 接收 token 并渲染
- **兼容性**：非流式 `/api/chat` 接口保持原有逻辑，不影响

### 引用来源展示

每次回答完成后自动展示引用来源，点击展开可查看具体原文片段：
- 检索时记录 `source` 元数据（文档名 + 片段序号）
- 回答完成后以折叠面板展示引用列表
- 内置示例文档 `example_doc.txt` 不可删除

### 多文档管理

右侧边栏新增文档管理面板，可以一站式管理向量库：
- **查看文档列表**：显示文档名、上传时间、片段数量
- **删除文档**：一键从向量库删除指定文档（内置示例除外）
- **自动刷新**：上传/删除后自动更新列表和系统状态

---

## 📂 项目结构

```
rag-demo/
├── app.py              # Flask 主程序（核心逻辑）
├── requirements.txt    # 依赖列表
├── .env.example        # 环境变量示例（复制为 .env 使用）
├── .gitignore          # Git 忽略文件
├── README.md           # 项目说明
├── data/
│   └── example_doc.txt # 内置示例文档
├── static/
│   └── style.css       # 页面样式
└── templates/
    └── index.html      # 问答页面
```

---

## 💡 常见问题

### Q: 启动时报错 `No module named 'xxx'`
A: 运行 `pip install -r requirements.txt` 安装依赖，建议 Python 版本 ≥ 3.10。

### Q: 问答返回空或向量库不生效？
A: 检查 `.env` 文件是否配置正确的 `API_KEY` 和 `BASE_URL`。

### Q: 回答质量不高？
A: 
- 上传更高质量的文档（内容连贯、信息密度高）
- 尝试调整 `app.py` 中的 `chunk_size`（默认 500）和 `k`（默认检索 3 条）
- 使用更强的模型（如 `gpt-4o`），在 `get_llm()` 修改 `model` 参数

### Q: 怎么在简历上描述这个项目？
> **项目名称**：基于 RAG 的智能问答系统  
> - 使用 Flask 构建轻量级 Web 服务，实现文档上传、分片、向量化和问答全流程
> - 基于 LangChain 集成 LLM，使用 ChromaDB 实现本地向量存储
> - 采用 OpenAI 兼容 API 架构，支持 OpenAI / DeepSeek / StepFun 等多种模型后端
> - 使用 SSE 实现流式输出（打字机效果），提升用户体验
> - 实现引用来源展示，每次回答自动关联原文片段，增强可追溯性
> - 多文档管理面板，支持列表查看和删除，完善向量库生命周期管理

---

## 🎯 已完成功能

- [x] 文档上传（txt / md）及自动分片入库
- [x] 文本分块向量库
- [x] RAG 问答（非流式）
- [x] 多轮对话（最多 20 条）
- [x] **流式输出（SSE / 打字机效果）**
- [x] **引用来源展示（点击展开看原文片段）**
- [x] **多文档管理（列表查看 / 删除文档）**
- [x] 系统状态实时显示

---

## 🚧 扩展建议

- [ ] 支持 PDF / Word 解析（引入 `PyPDF2` / `python-docx`）
- [ ] 多用户隔离（每人独立向量库）
- [ ] 接入本地模型（如 Ollama + Llama 3），完全离线运行
- [ ] 文档上传进度展示
- [ ] 引用来源高亮定位（跳转到原文对应位置）

---

*本项目仅用于学习和分享，请勿在生产环境直接使用。*
