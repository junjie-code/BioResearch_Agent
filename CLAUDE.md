# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioResearch Agent is a biological research assistant built with DeepSeek-V3 LLM and LangGraph. It uses a ReAct (Reasoning + Acting) pattern to autonomously plan tasks and invoke specialized tools for literature search, sequence analysis, cell image recognition, and internal knowledge base Q&A.

**Tech Stack:**
- LLM: DeepSeek-V3 (deepseek-chat)
- Agent Framework: LangGraph (ReAct pattern)
- Vector DB: Chroma
- Embedding: BAAI/bge-small-zh-v1.5
- Computer Vision: PyTorch U-Net
- Web UI: Streamlit

## Setup and Running

### Environment Setup
```bash
# Create .env file with API credentials
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start Streamlit web interface
streamlit run app.py
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_day1.py
python tests/test_day2.py
```

## Architecture

### Core Agent Flow (agent/graph.py)

The agent uses a LangGraph state machine with three nodes:

1. **agent_node**: LLM decides whether to call tools or answer directly
2. **tool_node_with_count**: Executes tools and increments call counter
3. **final_answer_node**: Forces summary when tool call limit is reached

**Flow:**
```
START → agent → [tools → agent]* → final_answer/END
```

**Key routing logic (should_continue):**
- If `tool_call_count >= MAX_AGENT_STEPS` → force summary via final_answer_node
- If LLM wants to call tools → go to tools node
- If LLM provides answer → END

**Important implementation details:**
- `_clean_messages()` removes orphaned AI tool_calls without corresponding tool responses (prevents DeepSeek API errors)
- `final_answer_node` uses LLM without tool binding to force text-only output
- Tool call limit (MAX_AGENT_STEPS) prevents infinite loops

### State Management (agent/state.py)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangGraph message history
    tool_call_count: int                     # Tracks tool invocations
```

### System Prompt (agent/prompts.py)

The SYSTEM_PROMPT defines:
- Available tools and their purposes
- Working principles (think before acting, use tools appropriately)
- Language rules: Keep literature results in English, use Chinese for summaries

**Critical rule:** If prompt contains >2 literature entries, agent should NOT call tools and answer directly.

### Tools (tools/)

All tools are registered in `tools/__init__.py` as `ALL_TOOLS` list. Each tool uses `@tool` decorator from LangChain.

**Available tools:**
1. **pubmed_search**: Query PubMed database for biomedical literature
2. **sequence_analysis**: Analyze DNA/protein sequences (GC content, composition, translation)
3. **cell_image_analysis**: U-Net-based cell nucleus detection and counting
4. **knowledge_qa**: RAG-based retrieval from internal knowledge base
5. **report_generator**: Generate structured reports from collected information

**RAG Knowledge Base (knowledge_qa.py):**
- Documents stored in `knowledge_base/docs/` (*.txt files)
- Vector DB persisted in `knowledge_base/chroma_db/`
- First run builds vector DB from documents
- Uses `_get_vectorstore()` singleton pattern to cache DB instance

**Cell Image Analysis (cell_image_analysis.py):**
- Uses pre-trained U-Net model from `models/unet_nuclei_epoch_20.pth`
- Model architecture must match training definition exactly
- Performs nucleus detection and counting on microscopy images

### Configuration (config/settings.py)

Key settings:
- `MAX_AGENT_STEPS = 1`: Maximum tool calls before forced summary (currently set to 1)
- `LLM_TEMPERATURE = 0.3`: Low temperature for agent scenarios to reduce hallucinations
- `DEEPSEEK_API_KEY`: Required environment variable
- Model paths: `UNET_WEIGHTS_PATH`, `CHROMA_DB_DIR`, `KNOWLEDGE_BASE_DIR`

### Web Interface (app.py)

Streamlit app with:
- Chat interface with message history
- Sidebar with example queries and tool list
- Image upload for cell analysis (saved to `uploads/` directory)
- Thread-based conversation memory using `thread_id`
- Real-time agent step visualization (tool calls, tool returns)

**Session state:**
- `messages`: Chat history
- `thread_id`: UUID for conversation persistence
- `uploaded_image_path`: Path to uploaded cell image

## Development Notes

### Adding New Tools

1. Create tool function in `tools/` with `@tool` decorator
2. Add to `ALL_TOOLS` list in `tools/__init__.py`
3. Update SYSTEM_PROMPT in `agent/prompts.py` to describe the new tool

### Modifying Agent Behavior

- **Change max tool calls**: Edit `MAX_AGENT_STEPS` in `config/settings.py`
- **Adjust LLM parameters**: Modify `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` in `config/settings.py`
- **Change routing logic**: Edit `should_continue()` function in `agent/graph.py`
- **Modify agent instructions**: Update `SYSTEM_PROMPT` in `agent/prompts.py`

### Knowledge Base Management

To add documents to the knowledge base:
1. Place `.txt` files in `knowledge_base/docs/`
2. Delete `knowledge_base/chroma_db/` directory to force rebuild
3. Restart the application

### Model Files

The U-Net model weights (`models/unet_nuclei_epoch_20.pth`) must be present for cell image analysis. The model architecture in `cell_image_analysis.py` must match the training architecture exactly.

## Common Issues

**DeepSeek API errors with tool calls:**
- The `_clean_messages()` function handles orphaned tool_calls
- Ensure tool responses always follow tool_calls in message history

**Agent loops infinitely:**
- Check `MAX_AGENT_STEPS` setting
- Verify `should_continue()` routing logic
- Ensure `final_answer_node` doesn't have tools bound

**Knowledge base not working:**
- Verify documents exist in `knowledge_base/docs/`
- Check if `chroma_db/` was created successfully
- Ensure embedding model downloads correctly (requires internet on first run)
