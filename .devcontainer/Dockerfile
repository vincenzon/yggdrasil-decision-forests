# syntax=docker/dockerfile:1
FROM ubuntu:latest
RUN userdel -r ubuntu

ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    build-essential \
    g++ \
    git \
    sudo \
    unzip \
    zip \
    wget \
    ca-certificates

# Install uv
RUN pip install --no-cache --break-system-packages uv

# Install Bazelisk (Bazel launcher)
RUN wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

# GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh

COPY postCreateCmd.sh /tmp
RUN chmod +x /tmp/postCreateCmd.sh

RUN addgroup --gid ${REMOTE_GID} ${REMOTE_USER}; \
    useradd \
      --create-home \
      --gid root \
      --groups sudo \
      --no-log-init \
      --shell /bin/bash \
      --system \
      --uid ${REMOTE_UID} \ 
      --gid ${REMOTE_GID} \
      ${REMOTE_USER}
      
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV HOME=/home/${REMOTE_USER}
USER ${REMOTE_USER}



