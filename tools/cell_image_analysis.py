# tools/cell_image_analysis.py
"""
细胞核图像分析工具

使用预训练的 Unet 模型检测和计数显微图像中的细胞核。
复用 nuclei-segmentation-unet 项目的模型权重。
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from langchain_core.tools import tool
from config.settings import UNET_WEIGHTS_PATH


# ===== U-Net 模型定义（与训练时完全一致）=====
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)

        b = self.bottleneck(p4)

        u1 = self.up1(b)
        c1 = self.conv1(torch.cat([d4, u1], dim=1))
        u2 = self.up2(c1)
        c2 = self.conv2(torch.cat([d3, u2], dim=1))
        u3 = self.up3(c2)
        c3 = self.conv3(torch.cat([d2, u3], dim=1))
        u4 = self.up4(c3)
        c4 = self.conv4(torch.cat([d1, u4], dim=1))

        return self.out(c4)


# ===== 预处理函数（与训练时完全一致）=====
def bio_preprocess(img):
    """中值滤波 + CLAHE 对比度增强"""
    blurred = cv2.medianBlur(img, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced


# ===== 模型单例缓存（避免每次调用都重新加载）=====
_model = None


def _get_model():
    """加载模型（只加载一次）"""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(UNET_WEIGHTS_PATH):
        raise FileNotFoundError(f"模型权重文件不存在: {UNET_WEIGHTS_PATH}")

    device = torch.device("cpu")
    _model = UNet(n_channels=1, n_classes=1).to(device)
    _model.load_state_dict(
        torch.load(UNET_WEIGHTS_PATH, map_location=device, weights_only=True)
    )
    _model.eval()
    print(f"[Unet] 模型加载成功: {UNET_WEIGHTS_PATH}")
    return _model


@tool
def cell_image_analysis(image_path: str) -> str:
    """
    分析显微图像中的细胞核，返回检测数量和分析结果。

    当用户需要分析细胞显微图像、检测细胞核数量、
    或评估细胞密度时使用此工具。

    Args:
        image_path: 细胞显微图像的文件路径（支持 .png, .jpg, .tif 格式）

    Returns:
        包含细胞核数量、面积统计等信息的分析报告
    """
    # === 1. 验证输入 ===
    if not os.path.exists(image_path):
        return f"错误：找不到图像文件 '{image_path}'。请提供正确的文件路径。"

    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if not image_path.lower().endswith(valid_extensions):
        return f"错误：不支持的图像格式。请使用以下格式：{valid_extensions}"

    try:
        # === 2. 读取图像（灰度）===
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            return f"错误：无法读取图像文件 '{image_path}'。文件可能已损坏。"

        original_h, original_w = original.shape

        # === 3. 预处理（与训练一致）===
        processed = bio_preprocess(original)
        processed_resized = cv2.resize(processed, (256, 256))

        # 转为 tensor: (1, 1, 256, 256)
        img_tensor = torch.from_numpy(
            processed_resized / 255.0
        ).float().unsqueeze(0).unsqueeze(0)

        # === 4. 模型推理 ===
        model = _get_model()
        with torch.no_grad():
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (pred_prob > 0.5).astype(np.uint8)

        # === 5. 连通域分析计数 ===
        labeled_array, num_features = ndimage.label(binary_mask)

        # 统计每个细胞核的面积，过滤噪点
        areas = []
        min_area = 10  # 面积小于10像素的视为噪点
        for i in range(1, num_features + 1):
            area = np.sum(labeled_array == i)
            if area >= min_area:
                areas.append(area)

        valid_count = len(areas)

        # === 6. 生成分析报告 ===
        report_lines = [
            "📊 细胞核图像分析报告",
            "=" * 40,
            f"图像路径: {image_path}",
            f"原始尺寸: {original_w} × {original_h} pixels",
            f"分析尺寸: 256 × 256 pixels",
            "",
            "📌 检测结果:",
            f"  检测到细胞核: {valid_count} 个",
            f"  (已过滤面积 < {min_area} 像素的噪点)",
        ]

        if areas:
            report_lines.extend([
                "",
                "📐 面积统计:",
                f"  平均面积: {np.mean(areas):.1f} pixels",
                f"  最大面积: {max(areas)} pixels",
                f"  最小面积: {min(areas)} pixels",
                f"  面积标准差: {np.std(areas):.1f} pixels",
                "",
                "📈 密度评估:",
                f"  掩膜覆盖率: {np.sum(binary_mask) / binary_mask.size * 100:.2f}%",
                f"  细胞密度: {valid_count / (256 * 256) * 10000:.2f} 个/万像素",
            ])

        # === 7. 保存预测结果图 ===
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs"
        )
        os.makedirs(output_dir, exist_ok=True)

        original_resized = cv2.resize(original, (256, 256))
        pred_visual = (binary_mask * 255).astype(np.uint8)
        combined = np.hstack([original_resized, pred_visual])

        save_name = f"analysis_{os.path.basename(image_path)}"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, combined)

        report_lines.extend([
            "",
            f"💾 预测结果已保存: {save_path}",
            "  (左：原图，右：细胞核预测掩膜)",
        ])

        return "\n".join(report_lines)

    except Exception as e:
        return f"图像分析出错: {type(e).__name__}: {str(e)}"