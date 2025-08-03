+++
title = 'RAG系统异步设计架构'
date = 2025-07-30T13:11:06+08:00
draft = true
+++

# RAG系统异步设计架构

## 目录
- [系统概览](#系统概览)
- [异步架构设计](#异步架构设计)
- [核心组件详解](#核心组件详解)
- [事件循环管理机制](#事件循环管理机制)
- [向量存储异步策略](#向量存储异步策略)
- [FastAPI集成模式](#fastapi集成模式)
- [LangGraph工作流设计](#langgraph工作流设计)
- [异步问题与解决方案](#异步问题与解决方案)
- [性能优化策略](#性能优化策略)
- [最佳实践](#最佳实践)

---

## 系统概览

### 整体架构图

![整体架构图.png](/images/RAG系统异步设计架构/整体架构图.png)

### 技术栈

- **Web框架**: FastAPI (异步ASGI)
- **异步运行时**: Python asyncio
- **向量数据库**: Milvus (支持异步操作)
- **LLM框架**: LangChain + LangGraph
- **文档处理**: LangChain Document Loaders + 模块化分块策略
- **嵌入模型**: DashScope Embeddings
- **分块策略**: 递归、Token、语义、字符、代码、格式特定策略
