# BioResearch Agent

基于 DeepSeek + LangGraph 的生物科研智能助手。通过 ReAct 模式自动规划任务，调用文献检索、序列分析、细胞图像识别、公司内部知识库问答等工具，完成复杂的生物科研辅助任务。

---

## 核心特性

- **智能任务规划**：基于 ReAct 模式自动拆解复杂任务，按步骤调用工具
- **上下文管理**：保留关键信息，通过智能摘要压缩上下文窗口，提升模型响应速度
- **知识库问答（RAG）**：从公司制度文件、操作手册等内部文档中检索答案
- **PubMed 文献检索**：根据需求检索全球最大的生物医学文献数据库
- **序列分析**：DNA/蛋白质序列的 GC 含量、碱基组成、翻译等分析
- **细胞核检测**：基于 U-Net 深度学习模型的细胞核自动检测与计数


---

## 系统架构
```
用户输入
   ↓
┌─────────────────────┐
│   LLM 决策层         │  ← System Prompt（任务规划策略）
│   (DeepSeek-V3)     │
└──────────┬──────────┘
           │ Function Calling
           ↓
┌─────────────────────┐     ┌────────────────────────────┐
│   工具调度层         │────→│  pubmed_search             │
│                     │────→│  sequence_analysis         │
│                     │────→│  cell_image_analysis (U-Net)│
│                     │────→│  knowledge_qa (RAG)        │
└──────────┬──────────┘     └────────────────────────────┘
           │
           ↓
┌─────────────────────┐
│   结果整合与展示     │ → Streamlit 界面
└─────────────────────┘
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 大语言模型 | DeepSeek-V3 (deepseek-chat) |
| Agent 框架 | LangGraph (ReAct 模式) |
| 向量数据库 | Chroma |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| 计算机视觉 | PyTorch U-Net |
| Web 框架 | Streamlit |

---

## 项目结构
```
BioResearch-Agent/
├── agent/                          # Agent 核心逻辑
│   ├── graph.py                    # LangGraph ReAct 循环实现
│   ├── state.py                    # 状态定义
│   ├── prompts.py                  # System Prompt 模板
│   └── context_manager.py          # 上下文管理（Token 优化）
├── tools/                          # 工具集
│   ├── pubmed_search.py            # PubMed API 文献检索
│   ├── sequence_analysis.py        # DNA/蛋白质序列分析
│   ├── cell_image_analysis.py      # U-Net 细胞核检测
│   ├── knowledge_qa.py             # RAG 知识库问答
│   └── report_generator.py         # 自动报告生成
├── knowledge_base/                 # 知识库
│   ├── docs/                       # 原始文档
│   └── vectorstore/                # 向量数据库
├── models/                         # 训练模型参数
│   └── unet_nuclei.pth             # U-Net 权重文件
├── config/                         # 配置文件
│   └── settings.py                 # 全局配置
├── tests/                          # 单元测试
├── app.py                          # Streamlit 主程序
├── requirements.txt                # 依赖列表
└── README.md                       # 项目文档
```
## 页面展示
<img width="2560" height="1279" alt="PixPin_2026-02-07_15-28-53" src="https://github.com/user-attachments/assets/abd2519f-4168-4e25-8589-ffa1e1ed61e7" />
