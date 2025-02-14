# Pythonのベースイメージを使用
FROM python:3.11-slim


# 作業ディレクトリを設定
WORKDIR /app

# Git LFSのインストール
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install

# リポジトリをクローン
RUN git clone https://huggingface.co/spaces/MakiAi/Style-Bert-VITS2-JVNV

# クローンしたリポジトリのディレクトリに移動
WORKDIR /app/Style-Bert-VITS2-JVNV

# Pythonの依存関係をインストール
RUN pip install -r requirements.txt
RUN pip install streamlit