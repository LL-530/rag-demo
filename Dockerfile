# 基于轻量 Python 3.11 镜像
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖（chromadb、某些依赖可能需要编译工具）
RUN apt-get update --no-install-recommends && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制依赖文件，利用 Docker 层缓存加速构建
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码
COPY . .

# 确保数据目录存在
RUN mkdir -p /app/data /app/chroma_db

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]
