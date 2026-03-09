"""
Image Agent：细胞核图像检测Agent
基于 U-Net 深度学习模型，接收 PIL Image，返回检测与计数结果
不调用LLM，纯本地推理
"""
import base64
import io
from PIL import Image
from langgraph.store.base import BaseStore

from tools.cell_image_analysis import detect_cells


def base64_to_pil(b64_string: str) -> Image.Image:
    """base64字符串 → PIL Image"""
    img_bytes = base64.b64decode(b64_string)
    img_buffer = io.BytesIO(img_bytes)
    pil_image = Image.open(img_buffer)
    return pil_image.convert("RGB")


async def image_agent_node(state: dict, *, store: BaseStore) -> dict:
    # 读取私有记忆
    namespace = ("memory", "image_agent")
    try:
        history = await store.aget(namespace, "analysis_history")
    except Exception:
        history = None

    # 提取图片
    image_b64 = _extract_image_base64(state["messages"])

    if not image_b64:
        return {"agent_results": {"image": "请上传需要分析的细胞图像"}}

    # U-Net 推理：base64 → PIL Image → detect_cells
    try:
        pil_image = base64_to_pil(image_b64)
        unet_result = detect_cells(pil_image)
    except Exception as e:
        return {"agent_results": {"image": f"图像检测出错：{str(e)}"}}

    # 格式化结果
    if isinstance(unet_result, dict):
        result_text = "\n".join(f"- {k}: {v}" for k, v in unet_result.items())
    else:
        result_text = str(unet_result)

    # 写入私有记忆
    from datetime import datetime
    record = {"time": datetime.now().isoformat(), "summary": result_text[:200]}
    updated = (history or []) + [record]
    try:
        await store.aput(namespace, "analysis_history", updated[-15:])
    except Exception:
        pass

    return {"agent_results": {"image": result_text}}


# ===== 消息解析工具 =====
def _extract_image_base64(messages: list) -> str | None:
    """从 messages 中提取纯 base64 字符串（不含 data:image 前缀）"""
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        return url.split(",", 1)[-1]
                    return url
    return None