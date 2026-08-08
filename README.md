# 📚 RAG Demo — 检索增强生成问答系统

> 大四毕设 / 简历项目 Demo  
> 技术栈：Flask + LangChain + ChromaDB + OpenAI 兼容 API

---

## 🌟 项目简介

RAG（Retrieval-Augmented Generation，检索增强生成）是目前最主流的 AI 应用架构之一。  
本项目实现了一个完整的 RAG 问答 Demo，涵盖：

| 步骤 | 说明 |
|------|------|
| 📄 文档上传 | 支持 `.txt` / `.md` 文件或直接粘贴文本 |
| ✂️ 文本切分 | 将长文本按段落切分为 500 字符的语义块 |
| 🔢 向量化 | 调用 OpenAI Embedding 生成向量 |
| 💾 持久化 | 向量存入本地 ChromaDB（无需数据库） |
| 🔍 问答检索 | 用户提问 → 检索 top-3 相关片段 |
| 🤖 LLM 生成 | 结合检索结果调用 LLM 生成结构化回答 |
| 💬 对话历史 | 浏览器 Session 内保留最近 20 轮对话 |
| ⚡ **流式输出** | **逐字显示回答，打字机效果** |
| 📌 **引用来源** | **回答末尾展示参考文档，点击展开查看原文片段** |
| 📂 **文档管理** | **列出所有已上传文档，支持删除管理** |

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

打开 `.env`，填写你的 API Key 和 Base URL：

```ini
API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BASE_URL=https://api.openai.com/v1
```

> **配置说明**
>
> - **使用 OpenAI 官方接口**：只需填写 `API_KEY`，`BASE_URL` 保持默认
> - **使用第三方中转服务**（DeepSeek / OneAPI / OpenRouter 等）：
>   将 `BASE_URL` 改为对应服务的 API 地址，如 `https://api.deepseek.com/v1`

### 3. 启动服务

```bash
python app.py
```

看到以下日志即表示启动成功：

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

首次启动时，系统会自动将 `data/example_doc.txt`（人工智能发展简史）写入向量库，打开页面即可直接提问。

---

## ✨ 新功能详解

### ⚡ 流式输出（打字机效果）

助手回答采用 **Server-Sent Events (SSE)** 实现实时逐字推送，配合前端 `ReadableStream` 逐字渲染，体验流畅丝滑。

- **后端**：新增 `/api/chat-stream` 接口，使用 `Response + generator` 流式返回
- **前端**：使用 `fetch + ReadableStream` 逐 token 接收并显示
- **兼容降级**：非流式 `/api/chat` 接口保持原有逻辑，不变

### 📌 引用来源展示

每次回答后自动展示「参考来源」区块，点击可展开查看具体原文片段。

- 检索时记录 `source` 元数据（文档文件名 + 片段序号）
- 回答完成后以美观的卡片形式展示来源列表
- 内置示例文档 `example_doc.txt` 受保护，不可删除

### 📂 多文档管理

左侧边栏新增「文档管理」面板，可以一览向量库中的所有文档。

- **查看文档列表**：显示文档名、上传时间、片段数量
- **删除文档**：一键从向量库移除对应数据（内置示例除外）
- 删除后自动刷新列表和系统状态

---

## 📁 项目结构

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

## ❓ 常见问题

### Q：启动时报错 `No module named 'xxx'`
A：运行 `pip install -r requirements.txt` 安装依赖，确认 Python 版本 ≥ 3.10。

### Q：问答返回"模型或向量库未就绪"
A：检查 `.env` 文件是否已配置正确的 `API_KEY` 和 `BASE_URL`。

### Q：回答质量不佳
A：
- 上传更高质量的文档（内容连贯、信息密度高）
- 尝试修改 `app.py` 中的 `chunk_size`（默认 500）和 `k`（默认检索 3 段）
- 使用更强的模型（如 `gpt-4o`），在 `get_llm()` 中修改 model 参数

### Q：如何在简历上描述这个项目？

> **项目名称**：基于 RAG 的智能文档问答系统
>
> - 使用 Flask 构建轻量 Web 服务，实现文档上传、切分、向量化及问答全流程
> - 基于 LangChain 框架接入 LLM，使用 ChromaDB 实现本地向量持久化存储
> - 采用 OpenAI 兼容 API 设计，支持 OpenAI / DeepSeek 等多种模型后端
> - **使用 SSE 实现流式问答（打字机效果），提升用户体验**
> - **实现引用来源展示，每个回答附上参考文档片段，增强可信度**
> - **多文档管理面板，支持列表展示与删除，完善向量库运维能力**
> - 实现上下文检索 + Prompt 工程，问答准确率达到 XX%（可以实测后填入）

---

## 🧠 技术要点（面试加分项）

- **为什么用 RAG 而不是直接问 LLM？** 解决大模型幻觉、知识时效性、私有数据访问问题
- **ChromaDB 为什么适合本地开发？** 零配置、纯 Python、数据持久化到本地文件
- **文本切分策略**：RecursiveCharacterTextSplitter 按段落、句号、空格递归切分，保留上下文
- **Top-K 检索**：每次检索最相关的 K 个片段，越多越全但速度越慢，需权衡
- **Prompt 设计**：要求 LLM 基于资料回答，回答不了就诚实说，减少幻觉
- **SSE 流式输出**：`Flask Response + generator` 后端推送，`fetch + ReadableStream` 前端消费
- **来源元数据追踪**：切分时附带 `source` 字段，去重聚合展示文档列表

---

## ✅ 已完成的功能

- [x] 文档上传（txt / md）及自动切分向量化
- [x] 文本粘贴入库
- [x] RAG 问答（非流式）
- [x] 对话历史（最近 20 轮）
- [x] **流式输出（SSE / 打字机效果）**
- [x] **引用来源展示（点击展开原文片段）**
- [x] **多文档管理（列表查看 / 删除文档）**
- [x] 系统状态面板实时刷新

---

## 🚧 扩展建议

- [ ] 支持 PDF / Word 解析（引入 `PyPDF2`、`python-docx`）
- [ ] 多用户隔离（各自拥有独立向量库）
- [ ] 接入开源模型（如 Ollama + Llama 3），完全离线运行
- [ ] 文档上传进度条显示
- [ ] 引用来源高亮定位（跳转到原文档对应位置）

---

*本项目仅用于学习与演示，请勿在生产环境直接使用。*
