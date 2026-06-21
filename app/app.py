"""
道路损伤检测系统 - Streamlit 可视化工作台
"""
import io
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parents[1] / "Ultralytics"),
)

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

matplotlib.use("Agg")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
WEIGHTS_DIR = APP_DIR / "weights"
DEMO_DIR = APP_DIR / "test_images"

CLASS_NAMES = {
    0: ("D00", "纵向裂缝", "Longitudinal Crack", (224, 72, 72)),
    1: ("D10", "横向裂缝", "Transverse Crack", (20, 148, 136)),
    2: ("D20", "龟裂/网状裂缝", "Alligator Crack", (245, 158, 11)),
    3: ("D40", "坑洞", "Pothole", (79, 70, 229)),
}

CLASS_COLOR = {v[0]: v[3] for v in CLASS_NAMES.values()}
MODEL_PATHS = {
    "YOLOv8n - 快速检测": WEIGHTS_DIR / "yolov8n_best.pt",
    "YOLOv8s - 精度优先": WEIGHTS_DIR / "yolov8s_best.pt",
}
TRAINING_CSVS = {
    "YOLOv8n": APP_DIR / "training_results.csv",
    "YOLOv8s": APP_DIR / "training_results_yolov8s.csv",
}
MODEL_TO_TRAINING = {
    "YOLOv8n - 快速检测": "YOLOv8n",
    "YOLOv8s - 精度优先": "YOLOv8s",
}
MODEL_META = {
    "YOLOv8n": {"参数量": "3.0M", "推理特点": "速度优先", "推荐场景": "批量筛查、低算力设备"},
    "YOLOv8s": {"参数量": "11.1M", "推理特点": "精度优先", "推荐场景": "重点路段复核、质量验收"},
}


st.set_page_config(
    page_title="道路损伤智能检测平台",
    page_icon="road",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --app-primary: #2563eb;
        --app-ok: #059669;
        --app-warn: #d97706;
        --app-danger: #dc2626;
    }

    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    }

    .stApp {
        --app-text: currentColor;
        --app-muted: color-mix(in srgb, currentColor 66%, transparent);
        --app-muted-strong: color-mix(in srgb, currentColor 82%, transparent);
        --app-panel: color-mix(in srgb, currentColor 4%, transparent);
        --app-panel-strong: color-mix(in srgb, currentColor 7%, transparent);
        --app-panel-soft: color-mix(in srgb, currentColor 8%, transparent);
        --app-line: color-mix(in srgb, currentColor 18%, transparent);
        --app-line-strong: color-mix(in srgb, currentColor 28%, transparent);
        --app-sidebar: color-mix(in srgb, currentColor 5%, transparent);
        --app-sidebar-text: currentColor;
        --app-sidebar-muted: color-mix(in srgb, currentColor 68%, transparent);
        --app-upload-bg: color-mix(in srgb, currentColor 6%, transparent);
        --app-ok-soft: color-mix(in srgb, var(--app-ok) 18%, transparent);
        --app-warn-soft: color-mix(in srgb, var(--app-warn) 18%, transparent);
        --app-danger-soft: color-mix(in srgb, var(--app-danger) 18%, transparent);
        --app-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        background: transparent;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }

    [data-testid="stSidebar"] {
        background: var(--app-sidebar);
        border-right: 1px solid var(--app-line);
    }

    [data-testid="stSidebar"] * {
        color: var(--app-sidebar-text) !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: var(--app-text) !important;
    }

    [data-testid="stSidebar"] .stSlider p {
        color: var(--app-sidebar-text) !important;
    }

    [data-testid="stToolbar"] [data-testid="stDeployButton"],
    [data-testid="stToolbar"] button[title="Deploy"],
    [data-testid="stToolbar"] button[aria-label="Deploy"],
    [data-testid="stDecoration"] button[title="Deploy"],
    [data-testid="stDecoration"] button[aria-label="Deploy"] {
        display: none !important;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 20px 24px;
        border: 1px solid var(--app-line);
        border-radius: 8px;
        background: var(--app-panel);
        box-shadow: var(--app-shadow);
        margin-bottom: 18px;
    }

    .topbar h1 {
        margin: 0;
        font-size: 1.65rem;
        line-height: 1.2;
        letter-spacing: 0;
        color: var(--app-text);
    }

    .topbar p {
        margin: 6px 0 0;
        color: var(--app-muted);
        font-size: 0.92rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border: 1px solid color-mix(in srgb, var(--app-ok) 36%, transparent);
        border-radius: 999px;
        background: var(--app-ok-soft);
        color: var(--app-muted-strong);
        font-size: 0.82rem;
        white-space: nowrap;
    }

    .section {
        padding: 18px;
        border: 1px solid var(--app-line);
        border-radius: 8px;
        background: var(--app-panel);
        box-shadow: var(--app-shadow);
        margin-bottom: 16px;
    }

    .section-title {
        margin: 0 0 12px;
        font-size: 1rem;
        font-weight: 700;
        color: var(--app-text);
    }

    .subtle {
        color: var(--app-muted);
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }

    .metric-tile {
        padding: 16px;
        border: 1px solid var(--app-line);
        border-radius: 8px;
        background: var(--app-panel);
        box-shadow: var(--app-shadow);
    }

    .metric-label {
        color: var(--app-muted);
        font-size: 0.78rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--app-text);
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.1;
    }

    .metric-note {
        color: var(--app-muted);
        font-size: 0.78rem;
        margin-top: 8px;
    }

    .metric-ok { border-left: 4px solid var(--app-ok); }
    .metric-warn { border-left: 4px solid var(--app-warn); }
    .metric-danger { border-left: 4px solid var(--app-danger); }
    .metric-primary { border-left: 4px solid var(--app-primary); }

    .score-panel {
        display: grid;
        gap: 12px;
    }

    .score-number {
        font-size: 2.6rem;
        line-height: 1;
        font-weight: 800;
        color: var(--app-text);
    }

    .score-bar {
        height: 12px;
        overflow: hidden;
        border-radius: 999px;
        background: var(--app-panel-soft);
        border: 1px solid var(--app-line);
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--app-danger), var(--app-warn), var(--app-ok));
    }

    .advice-list {
        display: grid;
        gap: 10px;
        margin-top: 8px;
    }

    .advice-item {
        padding: 10px 12px;
        border: 1px solid var(--app-line);
        border-radius: 8px;
        background: var(--app-panel-soft);
        color: var(--app-muted-strong);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .class-list {
        display: grid;
        gap: 8px;
    }

    .class-chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 9px 10px;
        border: 1px solid var(--app-line);
        border-radius: 8px;
        background: var(--app-panel-soft);
        font-size: 0.85rem;
    }

    .class-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .color-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    .image-caption {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding: 8px 2px 10px;
        font-size: 0.86rem;
        color: var(--app-muted);
    }

    .empty-card {
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 18px;
        border: 1px dashed var(--app-line-strong);
        border-radius: 8px;
        background: var(--app-panel-soft);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1px solid var(--app-line-strong) !important;
        color: var(--app-text) !important;
        background: var(--app-panel) !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: var(--app-primary) !important;
        border-color: var(--app-primary) !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 8px !important;
        border: 1px dashed var(--app-line-strong) !important;
        background: var(--app-upload-bg) !important;
        color: var(--app-text) !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] * {
        color: inherit;
    }

    .stTabs [role="tab"] {
        font-weight: 700;
    }

    @media (max-width: 900px) {
        .topbar {
            align-items: flex-start;
            flex-direction: column;
        }
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 560px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_key: str):
    model_path = MODEL_PATHS[model_key]
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}")
    return YOLO(str(model_path))


def load_font(size: int):
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_detections(image_rgb: np.ndarray, detections: list[dict]) -> Image.Image:
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    font = load_font(15)
    small_font = load_font(13)
    line_width = max(2, int(min(img.size) / 360))

    for det in detections:
        code = det["类别编号"]
        color = CLASS_COLOR.get(code, (37, 99, 235))
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = f"{code} {det['置信度']:.2f}"
        try:
            text_box = draw.textbbox((0, 0), label, font=small_font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
        except Exception:
            text_w = len(label) * 8
            text_h = 16

        label_y = max(0, y1 - text_h - 8)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 7], fill=color)
        draw.text((x1 + 5, label_y + 3), label, fill=(255, 255, 255), font=small_font)

    return img


def run_inference(model, image_bgr: np.ndarray, conf: float):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    results = model(image_bgr, conf=conf, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = xyxy[idx]
            code, zh_name, en_name, _ = CLASS_NAMES[cls_id]
            detections.append(
                {
                    "序号": idx + 1,
                    "类别编号": code,
                    "类别名称": zh_name,
                    "英文名称": en_name,
                    "置信度": round(conf_val, 3),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "中心X": int((x1 + x2) / 2),
                    "中心Y": int((y1 + y2) / 2),
                    "宽度": int(x2 - x1),
                    "高度": int(y2 - y1),
                }
            )

    return image_rgb, draw_detections(image_rgb, detections), detections, elapsed_ms


def get_demo_images():
    if not DEMO_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in DEMO_DIR.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def summarize_detections(detections: list[dict]) -> pd.DataFrame:
    base_rows = [
        {"类别编号": code, "类别名称": zh_name, "数量": 0, "平均置信度": 0.0}
        for code, zh_name, _, _ in CLASS_NAMES.values()
    ]
    if not detections:
        return pd.DataFrame(base_rows)

    df = pd.DataFrame(detections)
    summary = (
        df.groupby(["类别编号", "类别名称"])
        .agg(数量=("置信度", "count"), 平均置信度=("置信度", "mean"))
        .reset_index()
    )
    summary["平均置信度"] = summary["平均置信度"].round(3)
    summary = pd.concat([pd.DataFrame(base_rows), summary], ignore_index=True)
    summary = summary.groupby(["类别编号", "类别名称"], as_index=False).agg(
        数量=("数量", "max"),
        平均置信度=("平均置信度", "max"),
    )
    return summary.sort_values("类别编号")


def evaluate_road(detections: list[dict]):
    counts = {code: 0 for code, _, _, _ in CLASS_NAMES.values()}
    for det in detections:
        counts[det["类别编号"]] += 1

    if counts["D40"] > 0 or counts["D20"] >= 3:
        return "重度损伤", "建议优先巡检坑洞和大面积龟裂区域", "metric-danger", counts
    if detections:
        return "存在损伤", "建议复核裂缝位置并纳入维护计划", "metric-warn", counts
    return "未检出损伤", "当前阈值下未发现明显损伤", "metric-ok", counts


def calculate_health_score(counts: dict[str, int]) -> tuple[int, str, str]:
    weights = {"D00": 6, "D10": 5, "D20": 10, "D40": 18}
    total = sum(counts.values())
    penalty = sum(counts.get(code, 0) * weight for code, weight in weights.items())
    penalty += max(0, total - 4) * 2
    if counts.get("D40", 0) > 0:
        penalty += 8
    if counts.get("D20", 0) >= 3:
        penalty += 8

    score = int(max(0, min(100, 100 - penalty)))
    if score >= 85:
        return score, "健康", "metric-ok"
    if score >= 70:
        return score, "轻微损伤", "metric-warn"
    if score >= 50:
        return score, "中度损伤", "metric-warn"
    return score, "重度损伤", "metric-danger"


def generate_maintenance_advice(counts: dict[str, int], score: int) -> list[str]:
    total = sum(counts.values())
    if total == 0:
        return [
            "当前阈值下未检出明显道路损伤，建议保持常规巡检频率。",
            "若图像存在逆光、模糊或遮挡，可降低置信度阈值后复核。",
        ]

    advice = []
    if counts.get("D40", 0):
        advice.append("检测到坑洞 D40，建议优先安排现场核查，并进行坑槽修补或临时安全围挡。")
    if counts.get("D20", 0):
        advice.append("检测到龟裂/网状裂缝 D20，建议评估基层承载情况，必要时进行铣刨重铺或局部加固。")
    if counts.get("D00", 0):
        advice.append("检测到纵向裂缝 D00，建议进行灌缝封闭，防止雨水下渗扩大病害。")
    if counts.get("D10", 0):
        advice.append("检测到横向裂缝 D10，建议复核裂缝宽度和延展方向，纳入周期性养护计划。")
    if score < 60:
        advice.append("道路健康评分偏低，建议将该路段列为近期重点养护对象。")
    elif score < 80:
        advice.append("道路存在可见损伤，建议建立复查记录并跟踪病害发展趋势。")
    else:
        advice.append("总体健康状态尚可，建议以预防性养护和定期巡检为主。")
    return advice


def crop_detection_regions(image_rgb: np.ndarray, detections: list[dict], padding_ratio: float = 0.22):
    crops = []
    height, width = image_rgb.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad = int(max(box_w, box_h) * padding_ratio)
        left = max(0, x1 - pad)
        top = max(0, y1 - pad)
        right = min(width, x2 + pad)
        bottom = min(height, y2 + pad)
        if right <= left or bottom <= top:
            continue
        crop = Image.fromarray(image_rgb[top:bottom, left:right])
        crops.append(
            {
                "title": f"{det['序号']}. {det['类别编号']} {det['类别名称']}  置信度 {det['置信度']:.2f}",
                "image": crop,
            }
        )
    return crops


def render_metric(label: str, value: str, note: str, tone: str = "metric-primary"):
    st.markdown(
        f"""
        <div class="metric-tile {tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar():
    st.markdown(
        """
        <div class="topbar">
            <div>
                <h1>道路损伤智能检测平台</h1>
                <p>基于 YOLOv8 与 RDD2022 数据集，面向路面裂缝、龟裂和坑洞的图像检测工作台。</p>
            </div>
            <div class="status-pill">本地推理服务就绪</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 8px 0 16px;">
                <div style="font-size:1.15rem;font-weight:800;">RoadDamage AI</div>
                <div style="font-size:0.78rem;color:var(--app-sidebar-muted) !important;margin-top:4px;">
                    YOLOv8 道路病害检测
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 检测设置")
        model_choice = st.selectbox("模型权重", list(MODEL_PATHS.keys()), index=0)
        conf_threshold = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)

        st.markdown("### 检测类别")
        st.markdown('<div class="class-list">', unsafe_allow_html=True)
        for code, zh_name, en_name, color in CLASS_NAMES.values():
            r, g, b = color
            st.markdown(
                f"""
                <div class="class-chip">
                    <span class="class-left">
                        <span class="color-dot" style="background:rgb({r},{g},{b});"></span>
                        <span>{code} {zh_name}</span>
                    </span>
                    <span style="font-size:0.74rem;color:var(--app-sidebar-muted) !important;">{en_name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 环境状态")
        st.caption("PyTorch 2.6 · CUDA 12.4 · Ultralytics 8.4")

    return model_choice, conf_threshold


def render_input_area():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">输入图像</div>', unsafe_allow_html=True)

    input_col, action_col = st.columns([4, 1])
    with input_col:
        uploaded_files = st.file_uploader(
            "上传路面图片",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    with action_col:
        use_demo = st.button("随机演示图", width="stretch", type="primary")

    st.markdown(
        '<div class="subtle">支持 JPG、JPEG、PNG，可一次选择多张图片。上传图片优先；未上传时可使用内置演示图进行快速检测。</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    image_items = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                image_items.append(
                    {
                        "label": uploaded_file.name,
                        "image": Image.open(uploaded_file).convert("RGB").copy(),
                        "source": "upload",
                    }
                )
            except Exception as exc:
                st.warning(f"{uploaded_file.name} 读取失败: {exc}")
    elif use_demo:
        demo_images = get_demo_images()
        if demo_images:
            demo_path = random.choice(demo_images)
            image_items.append(
                {
                    "label": demo_path.name,
                    "image": Image.open(demo_path).convert("RGB").copy(),
                    "source": "demo",
                }
            )

    return image_items


def build_batch_outputs(model, image_items: list[dict], conf_threshold: float):
    results = []
    progress = st.progress(0.0, text="准备批量检测...")
    total = len(image_items)

    for index, item in enumerate(image_items, start=1):
        progress.progress((index - 1) / total, text=f"正在检测 {index}/{total}: {item['label']}")
        image_array = np.array(item["image"].convert("RGB"))
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_rgb, annotated_image, detections, elapsed_ms = run_inference(
            model, image_bgr, conf_threshold
        )
        road_status, suggestion, tone, counts = evaluate_road(detections)
        health_score, health_level, health_tone = calculate_health_score(counts)
        advice = generate_maintenance_advice(counts, health_score)
        avg_conf = np.mean([det["置信度"] for det in detections]) if detections else 0.0
        results.append(
            {
                "image_index": index,
                "label": item["label"],
                "source": item["source"],
                "width": int(image_rgb.shape[1]),
                "height": int(image_rgb.shape[0]),
                "image_rgb": image_rgb,
                "annotated_image": annotated_image,
                "detections": detections,
                "elapsed_ms": elapsed_ms,
                "road_status": road_status,
                "suggestion": suggestion,
                "tone": tone,
                "health_score": health_score,
                "health_level": health_level,
                "health_tone": health_tone,
                "advice": advice,
                "counts": counts,
                "avg_conf": avg_conf,
            }
        )

    progress.progress(1.0, text=f"批量检测完成，共 {total} 张图片")
    progress.empty()
    return results


def build_batch_summary_df(batch_results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in batch_results:
        row = {
            "图片": item["label"],
            "宽度": item["width"],
            "高度": item["height"],
            "检测目标数": len(item["detections"]),
            "平均置信度": round(item["avg_conf"], 3),
            "检测耗时(ms)": round(item["elapsed_ms"], 1),
            "路面评估": item["road_status"],
            "健康评分": item["health_score"],
            "养护等级": item["health_level"],
        }
        for code, _, _, _ in CLASS_NAMES.values():
            row[code] = item["counts"].get(code, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_batch_report_df(batch_results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in batch_results:
        common = {
            "图片序号": item["image_index"],
            "图片名称": item["label"],
            "图片来源": item["source"],
            "图片宽度": item["width"],
            "图片高度": item["height"],
            "检测耗时(ms)": round(item["elapsed_ms"], 1),
            "图片检测目标数": len(item["detections"]),
            "图片平均置信度": round(item["avg_conf"], 3),
            "路面评估": item["road_status"],
            "健康评分": item["health_score"],
            "养护等级": item["health_level"],
            "养护建议": "；".join(item["advice"]),
        }
        if item["detections"]:
            for det in item["detections"]:
                rows.append(
                    {
                        **common,
                        "目标序号": det["序号"],
                        "类别编号": det["类别编号"],
                        "类别名称": det["类别名称"],
                        "英文名称": det["英文名称"],
                        "置信度": det["置信度"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                        "中心X": det["中心X"],
                        "中心Y": det["中心Y"],
                        "宽度": det["宽度"],
                        "高度": det["高度"],
                    }
                )
        else:
            rows.append(
                {
                    **common,
                    "目标序号": "",
                    "类别编号": "",
                    "类别名称": "",
                    "英文名称": "",
                    "置信度": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "中心X": "",
                    "中心Y": "",
                    "宽度": "",
                    "高度": "",
                }
            )

    return pd.DataFrame(rows)


def ensure_history_state():
    st.session_state.setdefault("detection_history", [])
    st.session_state.setdefault("image_history", [])
    st.session_state.setdefault("history_run_keys", set())


def record_detection_history(batch_results: list[dict], model_choice: str, conf_threshold: float):
    ensure_history_state()
    if not batch_results:
        return

    run_key = "|".join(
        [
            model_choice,
            f"{conf_threshold:.2f}",
            *[
                f"{item['label']}:{item['width']}x{item['height']}:{len(item['detections'])}:{item['health_score']}"
                for item in batch_results
            ],
        ]
    )
    if run_key in st.session_state["history_run_keys"]:
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_images = len(batch_results)
    total_detections = sum(len(item["detections"]) for item in batch_results)
    damaged_images = sum(1 for item in batch_results if item["detections"])
    total_elapsed = sum(item["elapsed_ms"] for item in batch_results)
    avg_health = round(float(np.mean([item["health_score"] for item in batch_results])), 1)
    avg_conf = (
        round(float(np.mean([det["置信度"] for item in batch_results for det in item["detections"]])), 3)
        if total_detections
        else 0.0
    )

    st.session_state["detection_history"].append(
        {
            "批次ID": run_id,
            "检测时间": created_at,
            "模型": model_choice,
            "置信度阈值": conf_threshold,
            "图片数量": total_images,
            "检出图片数": damaged_images,
            "检测目标数": total_detections,
            "平均健康评分": avg_health,
            "平均置信度": avg_conf,
            "总耗时(ms)": round(total_elapsed, 1),
        }
    )

    for item in batch_results:
        image_row = {
            "批次ID": run_id,
            "检测时间": created_at,
            "模型": model_choice,
            "图片名称": item["label"],
            "图片来源": item["source"],
            "宽度": item["width"],
            "高度": item["height"],
            "检测目标数": len(item["detections"]),
            "平均置信度": round(item["avg_conf"], 3),
            "健康评分": item["health_score"],
            "养护等级": item["health_level"],
            "路面评估": item["road_status"],
            "检测耗时(ms)": round(item["elapsed_ms"], 1),
        }
        for code, _, _, _ in CLASS_NAMES.values():
            image_row[code] = item["counts"].get(code, 0)
        st.session_state["image_history"].append(image_row)

    st.session_state["history_run_keys"].add(run_key)


def render_detection_workspace(model_choice: str, conf_threshold: float):
    image_items = render_input_area()

    if not image_items:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">开始检测</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="empty-card">
                <div style="font-weight:700;margin-bottom:6px;">等待输入图像</div>
                <div class="subtle">请上传一张或多张路面图片，或点击“随机演示图”使用内置样例。检测结果会展示批量汇总、单图预览、损伤统计和 CSV 报告。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        demo_images = get_demo_images()[:8]
        if demo_images:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">演示图库</div>', unsafe_allow_html=True)
            demo_cols = st.columns(4)
            for index, image_path in enumerate(demo_images):
                with demo_cols[index % 4]:
                    st.image(str(image_path), caption=image_path.name, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        with st.spinner("正在加载模型并执行批量检测..."):
            model = load_model(model_choice)
            batch_results = build_batch_outputs(model, image_items, conf_threshold)
            record_detection_history(batch_results, model_choice, conf_threshold)
    except Exception as exc:
        st.error(f"检测失败: {exc}")
        return

    total_images = len(batch_results)
    total_detections = sum(len(item["detections"]) for item in batch_results)
    total_elapsed = sum(item["elapsed_ms"] for item in batch_results)
    damaged_images = sum(1 for item in batch_results if item["detections"])
    avg_conf = (
        np.mean([det["置信度"] for item in batch_results for det in item["detections"]])
        if total_detections
        else 0.0
    )
    avg_health = np.mean([item["health_score"] for item in batch_results]) if batch_results else 0.0

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric("图片数量", str(total_images), "本次批量检测输入", "metric-primary")
    with metric_cols[1]:
        render_metric("检测目标数", str(total_detections), "全部图片累计目标", "metric-primary")
    with metric_cols[2]:
        render_metric("平均健康评分", f"{avg_health:.0f}/100", f"平均置信度 {avg_conf:.1%}", "metric-ok")
    with metric_cols[3]:
        render_metric("总耗时", f"{total_elapsed:.0f} ms", f"{damaged_images} 张图片检出损伤", "metric-warn")
    st.markdown("</div>", unsafe_allow_html=True)

    summary_df = build_batch_summary_df(batch_results)
    report_df = build_batch_report_df(batch_results)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">批量检测汇总</div>', unsafe_allow_html=True)
    st.dataframe(summary_df, hide_index=True, width="stretch")

    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        "下载批量 CSV 检测报告",
        csv_buffer.getvalue(),
        "road_damage_batch_report.csv",
        "text/csv",
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    selected_index = 0
    if len(batch_results) > 1:
        selected_index = st.selectbox(
            "选择图片查看检测详情",
            list(range(len(batch_results))),
            format_func=lambda idx: f"{idx + 1}. {batch_results[idx]['label']}",
        )
    current = batch_results[selected_index]
    detections = current["detections"]

    score_fill_width = max(0, min(100, current["health_score"]))
    advice_html = "".join([f'<div class="advice-item">{item}</div>' for item in current["advice"]])
    health_col, advice_col = st.columns([1, 2])
    with health_col:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">道路健康评分</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="score-panel">
                <div>
                    <div class="score-number">{current["health_score"]}</div>
                    <div class="subtle">满分 100 · {current["health_level"]}</div>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width:{score_fill_width}%;"></div>
                </div>
                <div class="subtle">评分依据：损伤类别、数量和严重程度加权计算。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with advice_col:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">养护建议</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="advice-list">{advice_html}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="image-caption"><strong>原始图像</strong><span>{current["label"]}</span></div>',
            unsafe_allow_html=True,
        )
        st.image(current["image_rgb"], width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with img_col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="image-caption"><strong>检测结果</strong><span>阈值 {conf_threshold:.2f}</span></div>',
            unsafe_allow_html=True,
        )
        st.image(current["annotated_image"], width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    summary_df = summarize_detections(detections)
    stat_col, chart_col = st.columns([1, 1])
    with stat_col:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">损伤统计</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_col:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">类别分布</div>', unsafe_allow_html=True)
        chart_df = summary_df[["类别编号", "数量"]].set_index("类别编号")
        st.bar_chart(chart_df, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    if detections:
        crops = crop_detection_regions(current["image_rgb"], detections)
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">检测框局部放大</div>', unsafe_allow_html=True)
        crop_cols = st.columns(3)
        for index, crop in enumerate(crops[:12]):
            with crop_cols[index % 3]:
                st.image(crop["image"], caption=crop["title"], width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    if detections:
        detail_df = pd.DataFrame(detections)
        detail_df.insert(0, "图片名称", current["label"])
        visible_columns = ["图片名称", "序号", "类别编号", "类别名称", "置信度", "中心X", "中心Y", "宽度", "高度"]
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">当前图片检测明细</div>', unsafe_allow_html=True)
        st.dataframe(detail_df[visible_columns], hide_index=True, width="stretch")

        csv_buffer = io.StringIO()
        detail_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            "下载当前图片 CSV 明细",
            csv_buffer.getvalue(),
            "road_damage_report.csv",
            "text/csv",
            width="stretch",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("当前阈值下未检测到损伤。可以尝试降低置信度阈值，或更换更清晰的路面图像。")


def read_training_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_weight_path_for_model(model_name: str) -> Path:
    if model_name == "YOLOv8s":
        return WEIGHTS_DIR / "yolov8s_best.pt"
    return WEIGHTS_DIR / "yolov8n_best.pt"


def collect_model_metrics() -> pd.DataFrame:
    rows = []
    for model_name, csv_path in TRAINING_CSVS.items():
        df = read_training_csv(csv_path)
        weight_path = get_weight_path_for_model(model_name)
        meta = MODEL_META.get(model_name, {})
        if df is None or df.empty:
            rows.append(
                {
                    "模型": model_name,
                    "参数量": meta.get("参数量", "-"),
                    "权重大小(MB)": round(weight_path.stat().st_size / 1024 / 1024, 1)
                    if weight_path.exists()
                    else None,
                    "mAP50": None,
                    "mAP50-95": None,
                    "Precision": None,
                    "Recall": None,
                    "最佳Epoch": None,
                    "训练时长(h)": None,
                    "推理特点": meta.get("推理特点", "-"),
                    "推荐场景": meta.get("推荐场景", "-"),
                }
            )
            continue

        final = df.iloc[-1]
        best_map50_idx = df["metrics/mAP50(B)"].idxmax()
        rows.append(
            {
                "模型": model_name,
                "参数量": meta.get("参数量", "-"),
                "权重大小(MB)": round(weight_path.stat().st_size / 1024 / 1024, 1)
                if weight_path.exists()
                else None,
                "mAP50": round(float(final["metrics/mAP50(B)"]), 4),
                "mAP50-95": round(float(final["metrics/mAP50-95(B)"]), 4),
                "Precision": round(float(final["metrics/precision(B)"]), 4),
                "Recall": round(float(final["metrics/recall(B)"]), 4),
                "最佳mAP50": round(float(df["metrics/mAP50(B)"].max()), 4),
                "最佳Epoch": int(df.loc[best_map50_idx, "epoch"]),
                "训练时长(h)": round(float(df["time"].iloc[-1]) / 3600, 2),
                "推理特点": meta.get("推理特点", "-"),
                "推荐场景": meta.get("推荐场景", "-"),
            }
        )
    return pd.DataFrame(rows)


def get_theme_palette():
    theme = getattr(st.context, "theme", {}) or {}
    if hasattr(theme, "to_dict"):
        theme = theme.to_dict()
    elif not isinstance(theme, dict):
        theme = {}

    base = theme.get("base") or st.get_option("theme.base") or "light"
    is_dark = base == "dark"
    primary = theme.get("primaryColor") or "#2563eb"
    return {
        "is_dark": is_dark,
        "primary": primary,
        "text": "#e5e7eb" if is_dark else "#111827",
        "muted": "#94a3b8" if is_dark else "#64748b",
        "grid": "#334155" if is_dark else "#cbd5e1",
        "panel": "#0f172a" if is_dark else "#ffffff",
        "blue": "#60a5fa" if is_dark else "#2563eb",
        "green": "#34d399" if is_dark else "#059669",
        "orange": "#fbbf24" if is_dark else "#d97706",
        "violet": "#a78bfa" if is_dark else "#7c3aed",
    }


def style_training_axes(fig, axes, palette):
    fig.patch.set_alpha(0)
    for axis in axes:
        axis.set_facecolor("none")
        axis.grid(alpha=0.38, color=palette["grid"])
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["bottom"].set_color(palette["grid"])
        axis.spines["left"].set_color(palette["grid"])
        axis.tick_params(colors=palette["muted"])
        axis.xaxis.label.set_color(palette["muted"])
        axis.yaxis.label.set_color(palette["muted"])
        axis.title.set_color(palette["text"])
        axis.title.set_fontweight("bold")


def render_training_page():
    palette = get_theme_palette()

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">模型性能仪表盘</div>', unsafe_allow_html=True)
    model_name = st.selectbox("选择训练结果", list(TRAINING_CSVS.keys()), label_visibility="collapsed")
    df = read_training_csv(TRAINING_CSVS[model_name])
    if df is None:
        st.error(f"未找到训练结果文件: {TRAINING_CSVS[model_name]}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    meta = MODEL_META.get(model_name, {})
    weight_path = get_weight_path_for_model(model_name)
    weight_size = f"{weight_path.stat().st_size / 1024 / 1024:.1f} MB" if weight_path.exists() else "未找到"
    final = df.iloc[-1]
    best_map50 = float(df["metrics/mAP50(B)"].max())
    best_map50_ep = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])
    best_map95 = float(df["metrics/mAP50-95(B)"].max())

    result_cols = st.columns(6)
    with result_cols[0]:
        st.metric("mAP50", f"{final['metrics/mAP50(B)']:.4f}", f"Best {best_map50:.4f} @E{best_map50_ep}")
    with result_cols[1]:
        st.metric("mAP50-95", f"{final['metrics/mAP50-95(B)']:.4f}", f"Best {best_map95:.4f}")
    with result_cols[2]:
        st.metric("Precision", f"{final['metrics/precision(B)']:.4f}")
    with result_cols[3]:
        st.metric("Recall", f"{final['metrics/recall(B)']:.4f}")
    with result_cols[4]:
        st.metric("参数量", meta.get("参数量", "-"))
    with result_cols[5]:
        st.metric("权重大小", weight_size)

    st.caption(f"推理特点：{meta.get('推理特点', '-')} · 推荐场景：{meta.get('推荐场景', '-')}")

    st.markdown("</div>", unsafe_allow_html=True)

    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
    style_training_axes(fig1, axes, palette)

    axes[0].plot(df["epoch"], df["train/box_loss"], label="train box", color=palette["blue"])
    axes[0].plot(df["epoch"], df["train/cls_loss"], label="train cls", color=palette["green"])
    axes[0].plot(df["epoch"], df["train/dfl_loss"], label="train dfl", color=palette["orange"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(facecolor=palette["panel"], edgecolor=palette["grid"], labelcolor=palette["text"])

    axes[1].plot(df["epoch"], df["val/box_loss"], label="val box", color=palette["blue"])
    axes[1].plot(df["epoch"], df["val/cls_loss"], label="val cls", color=palette["green"])
    axes[1].plot(df["epoch"], df["val/dfl_loss"], label="val dfl", color=palette["orange"])
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(facecolor=palette["panel"], edgecolor=palette["grid"], labelcolor=palette["text"])
    style_training_axes(fig1, axes, palette)
    fig1.tight_layout()

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">损失曲线</div>', unsafe_allow_html=True)
    st.pyplot(fig1, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    style_training_axes(fig2, axes2, palette)

    axes2[0].plot(df["epoch"], df["metrics/mAP50(B)"], color=palette["blue"], linewidth=2)
    axes2[0].plot(df["epoch"], df["metrics/mAP50-95(B)"], color=palette["violet"], linewidth=2)
    axes2[0].set_title("mAP")
    axes2[0].set_xlabel("Epoch")
    axes2[0].legend(["mAP50", "mAP50-95"], facecolor=palette["panel"], edgecolor=palette["grid"], labelcolor=palette["text"])

    axes2[1].plot(df["epoch"], df["metrics/precision(B)"], color=palette["green"], linewidth=2)
    axes2[1].plot(df["epoch"], df["metrics/recall(B)"], color=palette["orange"], linewidth=2)
    axes2[1].set_title("Precision and Recall")
    axes2[1].set_xlabel("Epoch")
    axes2[1].legend(["Precision", "Recall"], facecolor=palette["panel"], edgecolor=palette["grid"], labelcolor=palette["text"])
    style_training_axes(fig2, axes2, palette)
    fig2.tight_layout()

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">精度指标</div>', unsafe_allow_html=True)
    st.pyplot(fig2, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_compare_page():
    metrics_df = collect_model_metrics()
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">模型对比页面</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">对比本地已训练的 YOLOv8n 与 YOLOv8s 权重，辅助选择“速度优先”或“精度优先”的检测模式。</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(metrics_df, hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    chart_cols = st.columns([1, 1])
    numeric_cols = ["mAP50", "mAP50-95", "Precision", "Recall"]
    chart_df = metrics_df.set_index("模型")[numeric_cols].apply(pd.to_numeric, errors="coerce")
    with chart_cols[0]:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">精度指标对比</div>', unsafe_allow_html=True)
        st.bar_chart(chart_df, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_cols[1]:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">模型体量对比</div>', unsafe_allow_html=True)
        size_df = metrics_df.set_index("模型")[["权重大小(MB)"]].apply(pd.to_numeric, errors="coerce")
        st.bar_chart(size_df, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">选择建议</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="advice-list">
            <div class="advice-item">若需要快速批量筛查大量图片，优先选择 YOLOv8n - 快速检测。</div>
            <div class="advice-item">若需要对重点路段、报告材料或疑似损伤图片进行更稳妥复核，优先选择 YOLOv8s - 精度优先。</div>
            <div class="advice-item">展示汇报时可先使用 YOLOv8n 体现响应速度，再切换 YOLOv8s 展示精度提升。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_page():
    ensure_history_state()
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">历史检测记录</div>', unsafe_allow_html=True)

    if not st.session_state["detection_history"]:
        st.markdown(
            '<div class="empty-card"><div style="font-weight:700;margin-bottom:6px;">暂无历史记录</div><div class="subtle">完成一次图片检测后，系统会自动记录本次批次摘要和每张图片的检测结果。</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    batch_df = pd.DataFrame(st.session_state["detection_history"])
    image_df = pd.DataFrame(st.session_state["image_history"])

    h_cols = st.columns(4)
    with h_cols[0]:
        st.metric("检测批次", len(batch_df))
    with h_cols[1]:
        st.metric("累计图片", int(batch_df["图片数量"].sum()))
    with h_cols[2]:
        st.metric("累计目标", int(batch_df["检测目标数"].sum()))
    with h_cols[3]:
        st.metric("平均健康评分", f"{batch_df['平均健康评分'].mean():.1f}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">批次历史</div>', unsafe_allow_html=True)
    st.dataframe(batch_df, hide_index=True, width="stretch")
    batch_csv = io.StringIO()
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")
    st.download_button(
        "下载批次历史 CSV",
        batch_csv.getvalue(),
        "road_damage_batch_history.csv",
        "text/csv",
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">图片级历史</div>', unsafe_allow_html=True)
    st.dataframe(image_df, hide_index=True, width="stretch")
    image_csv = io.StringIO()
    image_df.to_csv(image_csv, index=False, encoding="utf-8-sig")
    st.download_button(
        "下载图片级历史 CSV",
        image_csv.getvalue(),
        "road_damage_image_history.csv",
        "text/csv",
        width="stretch",
    )
    if st.button("清空当前会话历史记录", width="stretch"):
        st.session_state["detection_history"] = []
        st.session_state["image_history"] = []
        st.session_state["history_run_keys"] = set()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_data_page():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">数据集与模型说明</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="subtle">
        本系统使用 RDD2022 多国家道路损伤数据集，当前项目选择 Japan、India、Czech 三个国家的数据。
        标注从 XML 转换为 YOLO txt 格式，并按 8:1:1 划分为训练集、验证集和测试集。
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_root = ROOT_DIR / "datasets" / "RDD2022_Selected"
    rows = []
    for split in ["train", "val", "test"]:
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        rows.append(
            {
                "划分": split,
                "图片数量": len(list(image_dir.glob("*"))) if image_dir.exists() else 0,
                "标签数量": len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    model_rows = []
    for model_label, model_path in MODEL_PATHS.items():
        model_rows.append(
            {
                "模型": model_label,
                "权重文件": model_path.name,
                "文件大小": f"{model_path.stat().st_size / 1024 / 1024:.1f} MB"
                if model_path.exists()
                else "未找到",
            }
        )
    st.dataframe(pd.DataFrame(model_rows), hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    model_choice, conf_threshold = render_sidebar()
    render_topbar()

    tab_detect, tab_train, tab_compare, tab_history, tab_data = st.tabs(
        ["检测工作台", "模型性能仪表盘", "模型对比", "历史检测记录", "数据与模型"]
    )
    with tab_detect:
        render_detection_workspace(model_choice, conf_threshold)
    with tab_train:
        render_training_page()
    with tab_compare:
        render_model_compare_page()
    with tab_history:
        render_history_page()
    with tab_data:
        render_data_page()


if __name__ == "__main__":
    main()
