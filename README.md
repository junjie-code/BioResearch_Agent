# BioResearch Multi-Agent

基于 DeepSeek + LangGraph 的生物科研多智能体协作系统。通过 Supervisor 模式编排 3 个专业 Agent（文献检索、序列分析、细胞图像识别），实现复杂科研任务的自动拆解与协作完成。支持 Agent 级记忆隔离、MCP 工具解耦、微调 Qwen2-7B 恶意输入防御。


---

## 核心特性

- **多智能体协作**：基于 LangGraph Supervisor 模式，3 个专业 Agent 各司其职，Supervisor 自动拆解任务、编排执行顺序、汇总结果
- **三层记忆隔离**：Global State（全局共享）+ Scoped Store（Agent 私有记忆）+ Checkpointer（跨会话持久化），Agent 间记忆互不干扰
- **Agent 级上下文管理**：每个 Agent 只接收与自身相关的上下文，按关键词过滤历史消息，超长时自动摘要压缩，避免 token 浪费
- **MCP 工具解耦**：PubMed 文献检索通过 MCP Server 独立部署，Agent 与工具松耦合，支持热替换数据源
- **恶意输入防御**：基于微调 Qwen2-7B（LoRA）的二分类模型，识别提示词注入攻击，替代传统正则匹配，语义级防御
- **多模态图像分析**：基于 U-Net 深度学习模型的细胞核自动检测与计数，支持通过 Streamlit 上传图片
- **PubMed 文献检索**：根据需求检索全球最大的生物医学文献数据库
- **序列分析**：DNA/蛋白质序列的 GC 含量、碱基组成、翻译等分析

---

## 系统架构

```
用户输入（文本 / 图片）
        │
        ▼
┌───────────────────────────────────────────────┐
│              Prompt Guard                     │
│     微调 Qwen2-7B 恶意输入识别（LoRA）         │
└───────────────────┬───────────────────────────┘
                    │ 安全校验通过
                    ▼
          ┌──────────────────┐
          │    Supervisor     │  意图分类 + 任务拆解 + 结果汇总
          │   (DeepSeek-V3)  │  持有：全局摘要记忆
          └────────┬─────────┘
                   │ conditional_edges（支持多轮循环调度）
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
┌──────────┐┌─────────────┐┌────────────┐
│Literature││ Sequence    ││   Image    │
│  Agent   ││  Agent      ││   Agent    │
│ 文献检索  ││ 序列分析    ││ 细胞核检测  │
│          ││             ││            │
│MCP Server││Function call││  U-Net     │
│ PubMed   ││     NCBI     ││ (PyTorch)  │
└────┬─────┘└────┬─────────┘└─────┬──────┘
     │           │            │
     ▼           ▼            ▼
  PubMed      NCBI API    本地GPU推理
  (REST)     (Entrez)

每个 Agent 拥有独立的 Scoped Memory（私有记忆空间）
所有 Agent 通过 Global State 共享任务上下文
Checkpointer 提供跨会话持久化能力
```

---

## 记忆隔离架构

本项目解决多 Agent 共存时记忆互相污染的问题：

```
┌───────────────────────────────────────────────────┐
│   第1层：Global State（全局共享）                    │
│   task_id / user_intent / agent_results            │
├───────────────────────────────────────────────────┤
│   第2层：Scoped Store（Agent私有记忆）               │
│                                                    │
│   ("memory","lit_agent")  → 检索历史、论文缓存       │
│   ("memory","seq_agent")  → 序列查询缓存             │
│   ("memory","image_agent")→ 图像分析历史             │
│                                                    │
├───────────────────────────────────────────────────┤
│   第3层：Checkpointer（持久化）                      │
│   SqliteSaver，按 thread_id 隔离不同会话             │
└───────────────────────────────────────────────────┘
```


## 技术栈

| 组件 | 技术 |
|------|------|
| 大语言模型 | DeepSeek-V3 (deepseek-chat) |
| Agent 框架 | LangGraph (Supervisor 多Agent模式) |
| 工具协议 | MCP (Model Context Protocol) |
| 计算机视觉 | PyTorch U-Net |
| 安全防御 | 微调 Qwen2-7B (LoRA) 恶意分类 |
| Web 框架 | Streamlit |

---


## 项目结构

```
BioResearch_Agent/
├── agent/                              # Agent 核心逻辑
│   ├── graph.py                        # LangGraph 多Agent图定义（Supervisor模式）
│   ├── supervisor.py                   # Supervisor 意图分类 + 任务编排 + 结果汇总
│   ├── context_manager.py              # Agent级上下文裁剪与摘要压缩
│   ├── prompt_guard.py                 # 微调Qwen2-7B恶意输入识别
│   └── model_factory.py                # 分层模型管理工厂
│
├── agents/                             # 3个专业Agent
│   ├── literature_agent.py             # 文献检索Agent（MCP → PubMed）
│   ├── sequence_agent.py               # 序列分析Agent（MCP → NCBI）
│   └── image_agent.py                  # 图像分析Agent（U-Net细胞核检测）
│
├── mcp_servers/                        # MCP工具服务（独立进程）
│   └── pubmed_server.py                  # PubMed文献查询MCP Server
│
├── tools/                              # 底层工具实现
│   ├── pubmed_search.py                # PubMed API 调用
│   ├── sequence_analysis.py            # DNA/蛋白质序列分析
│   └── cell_image_analysis.py          # U-Net 细胞核检测
│
├── models/                             # 模型权重
│   ├── unet_nuclei.pth                 # U-Net 权重文件
│   └── prompt_guard_qwen2_7b/          # 微调Qwen2-7B分类模型
│
│
├── config/                             # 配置文件
│   └── settings.py                     # 全局配置
│
│
├── app.py                              # Streamlit 主程序
├── requirements.txt                    # 依赖列表
└── README.md                           # 项目文档
```

