"""
道路损伤检测系统 — 基于 YOLOv8 · 专业版
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import time
import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RoadDamage AI · 道路损伤检测",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 自定义 CSS ====================
st.markdown("""
<style>
    /* === 全局 === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* === 主背景渐变 === */
    .stApp {
        background: linear-gradient(135deg, #0f1729 0%, #1a2332 50%, #0f1729 100%);
    }

    /* === 侧边栏 === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }

    /* === 卡片容器 === */
    .card {
        background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* === 指标卡片 === */
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(99,102,241,0.02) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card.green {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.02) 100%);
        border-color: rgba(16,185,129,0.2);
    }
    .metric-card.amber {
        background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.02) 100%);
        border-color: rgba(245,158,11,0.2);
    }
    .metric-card.purple {
        background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(139,92,246,0.02) 100%);
        border-color: rgba(139,92,246,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* === 标题 === */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 4px;
    }

    /* === 检测标签 === */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-high { background: rgba(16,185,129,0.15); color: #10b981; }
    .badge-mid  { background: rgba(245,158,11,0.15); color: #f59e0b; }
    .badge-low  { background: rgba(239,68,68,0.15); color: #ef4444; }

    /* === 状态条 === */
    .status-bar {
        display: flex;
        gap: 8px;
        align-items: center;
        padding: 8px 16px;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.online { background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }

    /* === 分隔线 === */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 20px 0;
    }

    /* === 表格优化 === */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* === 按钮 === */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99,102,241,0.3);
    }

    /* === 文件上传 === */
    [data-testid="stFileUploader"] section {
        border-radius: 12px !important;
        border: 2px dashed rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.02) !important;
        transition: border-color 0.3s !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(99,102,241,0.4) !important;
    }

    /* === 标签页 === */
    .stTabs [role="tab"] {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 常量 ====================
CLASS_NAMES = {
    0: ("D00", "纵向裂缝", (255, 107, 107)),
    1: ("D10", "横向裂缝", (78, 205, 196)),
    2: ("D20", "龟裂/网状裂缝", (255, 230, 109)),
    3: ("D40", "坑洞", (163, 126, 240)),
}

CLASS_ZH = {v[0]: v[1] for v in CLASS_NAMES.values()}
CLASS_COLOR = {v[0]: v[2] for v in CLASS_NAMES.values()}

MODEL_PATHS = {
    "YOLOv8n (RDD 训练)": os.path.join(os.path.dirname(__file__), "weights", "yolov8n_best.pt"),
    "YOLOv8s (RDD 训练)": os.path.join(os.path.dirname(__file__), "weights", "yolov8s_best.pt"),
    "YOLOv8n (官方预训练)": "yolov8n.pt",
    "YOLOv8s (官方预训练)": "yolov8s.pt",
}

CSV_PATH = os.path.join(os.path.dirname(__file__), "training_results.csv")

# ==================== 模型 ====================
@st.cache_resource
def load_model(model_key: str):
    return YOLO(MODEL_PATHS[model_key])

# ==================== 自定义检测框渲染 ====================
def draw_detections_pil(image_rgb: np.ndarray, detections: list) -> Image.Image:
    """用 PIL 绘制高质量检测框，颜色按类别 + 置信度标注"""
    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)
    # 尝试加载字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 16)
                font_sm = ImageFont.truetype(fp, 13)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    line_w = max(2, int(min(img.size) / 400))

    for det in detections:
        code = det["类别编号"]
        conf = det["置信度"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color = CLASS_COLOR[code]

        # 外发光
        for offset in range(line_w + 1, 0, -1):
            alpha = int(60 * offset / (line_w + 1))
            glow_color = (*color, alpha)
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color, width=1,
            )

        # 主体框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        # 标签背景
        label = f"{code} {conf:.2f}"
        if font is not None:
            try:
                bbox = draw.textbbox((0, 0), label, font=font_sm)
            except Exception:
                bbox = (0, 0, len(label) * 8, 18)
        else:
            bbox = (0, 0, len(label) * 8, 18)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 4)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 4], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(0, 0, 0), font=font_sm)

    return img


def run_inference(model, image_bgr: np.ndarray, conf: float):
    """推理，返回 PIL 标注图 + 明细列表 + 耗时 + 原图 RGB"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    results = model(image_bgr, conf=conf, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000

    result = results[0]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = boxes_xyxy[i]
            detections.append({
                "序号": i + 1,
                "类别编号": CLASS_NAMES[cls_id][0],
                "类别名称": CLASS_NAMES[cls_id][1],
                "置信度": round(conf_val, 3),
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "中心X": int((x1 + x2) / 2),
                "中心Y": int((y1 + y2) / 2),
                "宽度": int(x2 - x1),
                "高度": int(y2 - y1),
            })

    annotated_pil = draw_detections_pil(image_rgb, detections)
    return image_rgb, annotated_pil, detections, elapsed


def confidence_badge(conf: float) -> str:
    if conf >= 0.7:
        return f'<span class="badge badge-high">● {conf:.3f}</span>'
    elif conf >= 0.4:
        return f'<span class="badge badge-mid">● {conf:.3f}</span>'
    return f'<span class="badge badge-low">● {conf:.3f}</span>'


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        <div style="font-size:2rem;">🛣️</div>
        <div>
            <div style="font-weight:700;font-size:1.1rem;color:#e2e8f0;">RoadDamage AI</div>
            <div style="font-size:0.7rem;color:#64748b;">YOLOv8 · RDD2022</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ 检测设置")
    model_choice = st.selectbox("模型选择", list(MODEL_PATHS.keys()), index=0)
    conf_threshold = st.slider(
        "置信度阈值", 0.05, 0.95, 0.25, 0.05,
        help="仅显示置信度高于此阈值的检测结果"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📋 检测类别")
    for cls_id, (code, name, color) in CLASS_NAMES.items():
        r, g, b = color
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
            f'<span style="width:10px;height:10px;border-radius:3px;'
            f'background:rgb({r},{g},{b});display:inline-block;"></span>'
            f'<code style="color:#c9d1d9;">{code}</code> '
            f'<span style="color:#94a3b8;font-size:0.85rem;">{name}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="status-bar">
        <span class="status-dot online"></span> GPU: RTX 4060 8GB
    </div>
    <div style="font-size:0.7rem;color:#475569;margin-top:6px;">
        PyTorch 2.6 · CUDA 12.4 · Ultralytics 8.4
    </div>
    """, unsafe_allow_html=True)

# ==================== Tab 结构 ====================
tab_detect, tab_train = st.tabs(["🔍 智能检测", "📈 训练分析"])

# ================================================================
#  Tab 1: 检测
# ================================================================
with tab_detect:
    # Hero
    st.markdown('<p class="hero-title">Road Damage Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">AI 驱动的路面损伤智能识别系统 · 支持 4 类道路损伤自动检测</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 上传 + 演示
    up_col, demo_col = st.columns([4, 1])
    with up_col:
        uploaded_file = st.file_uploader("拖拽或点击上传路面图片", type=["jpg", "jpeg", "png"])
    with demo_col:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("🎲 随机演示", width="stretch")

    image_source = None
    source_label = ""
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)
        source_label = uploaded_file.name
    elif use_demo:
        demo_dir = os.path.join(os.path.dirname(__file__), "test_images")
        demo_images = [f for f in os.listdir(demo_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if demo_images:
            import random
            random.seed()
            demo_file = random.choice(demo_images)
            image_source = Image.open(os.path.join(demo_dir, demo_file))
            source_label = demo_file

    if image_source is not None:
        img_array = np.array(image_source.convert("RGB"))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("🧠 AI 模型推理中..."):
            model = load_model(model_choice)
            img_rgb, annotated_pil, detections, elapsed_ms = run_inference(model, img_bgr, conf_threshold)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- 指标卡片 ----
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card purple">
                <div class="metric-value">{elapsed_ms:.0f}<span style="font-size:1rem;"> ms</span></div>
                <div class="metric-label">⚡ 推理耗时</div>
            </div>""", unsafe_allow_html=True)

        # 严重程度评估
        d00_cnt = sum(1 for d in detections if d["类别编号"] == "D00")
        d10_cnt = sum(1 for d in detections if d["类别编号"] == "D10")
        d20_cnt = sum(1 for d in detections if d["类别编号"] == "D20")
        d40_cnt = sum(1 for d in detections if d["类别编号"] == "D40")
        if d40_cnt > 0 or d20_cnt >= 3:
            severity, sev_color = "⚠️ 严重", "amber"
        elif len(detections) > 0:
            severity, sev_color = "📋 需关注", "purple"
        else:
            severity, sev_color = "✅ 良好", "green"

        with m2:
            st.markdown(f"""
            <div class="metric-card {sev_color}">
                <div class="metric-value">{severity}</div>
                <div class="metric-label">📊 路面评估</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(detections)}</div>
                <div class="metric-label">🎯 检测损伤数</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            avg_conf = np.mean([d["置信度"] for d in detections]) if detections else 0
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-value">{avg_conf:.2%}</div>
                <div class="metric-label">📈 平均置信度</div>
            </div>""", unsafe_allow_html=True)

        # ---- 图片对比 ----
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        img_c1, img_c2 = st.columns(2)

        with img_c1:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-weight:600;color:#e2e8f0;">📷 原始图像</span>
                <span style="font-size:0.75rem;color:#64748b;">{source_label}</span>
            </div>""", unsafe_allow_html=True)
            st.image(img_rgb, width="stretch")

        with img_c2:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-weight:600;color:#e2e8f0;">🔍 AI 检测结果</span>
                <span style="font-size:0.75rem;color:#64748b;">阈值: {conf_threshold:.2f}</span>
            </div>""", unsafe_allow_html=True)
            st.image(annotated_pil, width="stretch")

        # ---- 统计 ----
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if detections:
            s1, s2 = st.columns([1, 1])

            with s1:
                st.markdown("#### 📋 损伤统计")
                df = pd.DataFrame(detections)
                summary = (
                    df.groupby(["类别编号", "类别名称"])
                    .agg(数量=("置信度", "count"), 平均置信度=("置信度", "mean"))
                    .reset_index()
                )
                summary["平均置信度"] = summary["平均置信度"].round(3)
                # 确保 4 类都显示
                for c in CLASS_NAMES:
                    code, name, _ = CLASS_NAMES[c]
                    if code not in summary["类别编号"].values:
                        summary = pd.concat([
                            summary,
                            pd.DataFrame([{"类别编号": code, "类别名称": name, "数量": 0, "平均置信度": 0.0}]),
                        ], ignore_index=True)
                summary = summary.sort_values("类别编号")
                st.dataframe(summary, width="stretch", hide_index=True)

            with s2:
                st.markdown("#### 📊 损伤分布")
                chart_data = {CLASS_NAMES[c][0]: 0 for c in CLASS_NAMES}
                for det in detections:
                    chart_data[det["类别编号"]] += 1
                chart_df = pd.DataFrame({
                    "类别": [f"{CLASS_NAMES[k][0]}\n{CLASS_NAMES[k][1]}" for k in CLASS_NAMES],
                    "数量": [chart_data[CLASS_NAMES[k][0]] for k in CLASS_NAMES],
                })
                st.bar_chart(chart_df.set_index("类别"), width="stretch")

            # ---- 详细列表 ----
            st.markdown("#### 📝 检测明细")
            display_df = df[["序号", "类别编号", "类别名称", "置信度", "中心X", "中心Y", "宽度", "高度"]].copy()
            st.dataframe(display_df, width="stretch", hide_index=True)

            # ---- 导出 ----
            e1, e2 = st.columns([1, 4])
            with e1:
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 导出 CSV 报告",
                    csv_buf.getvalue(),
                    "road_damage_report.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with e2:
                st.caption(f"共 {len(detections)} 条记录，可下载后用 Excel 打开")
        else:
            st.info("💡 未检测到损伤 — 尝试降低左侧置信度阈值")

    else:
        # 空状态
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("""
            <div class="card" style="text-align:center;">
                <div style="font-size:2.5rem;">📤</div>
                <div style="font-weight:600;color:#e2e8f0;margin:8px 0;">上传图片</div>
                <div style="font-size:0.85rem;color:#94a3b8;">支持 JPG / PNG 格式<br>任意分辨率路面照片</div>
            </div>""", unsafe_allow_html=True)
        with g2:
            st.markdown("""
            <div class="card" style="text-align:center;">
                <div style="font-size:2.5rem;">🧠</div>
                <div style="font-weight:600;color:#e2e8f0;margin:8px 0;">AI 推理</div>
                <div style="font-size:0.85rem;color:#94a3b8;">YOLOv8 深度学习模型<br>毫秒级自动检测</div>
            </div>""", unsafe_allow_html=True)
        with g3:
            st.markdown("""
            <div class="card" style="text-align:center;">
                <div style="font-size:2.5rem;">📊</div>
                <div style="font-weight:600;color:#e2e8f0;margin:8px 0;">查看报告</div>
                <div style="font-size:0.85rem;color:#94a3b8;">损伤统计 + 可视化<br>一键导出 CSV</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📸 演示图库")
        demo_dir = os.path.join(os.path.dirname(__file__), "test_images")
        if os.path.exists(demo_dir):
            demo_imgs = sorted([
                f for f in os.listdir(demo_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])[:8]
            cols = st.columns(4)
            for i, fname in enumerate(demo_imgs):
                with cols[i % 4]:
                    img = Image.open(os.path.join(demo_dir, fname))
                    st.image(img, caption=fname, width="stretch")

# ================================================================
#  Tab 2: 训练分析
# ================================================================
with tab_train:
    st.markdown('<p class="hero-title">Training Analytics</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">YOLOv8n · RDD2022 数据集 · 50 Epochs · RTX 4060 8GB</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not os.path.exists(CSV_PATH):
        st.error("未找到 training_results.csv")
    else:
        df = pd.read_csv(CSV_PATH)
        final = df.iloc[-1]
        best_map50 = df["metrics/mAP50(B)"].max()
        best_map50_ep = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])

        # ---- 成绩卡 ----
        cm1, cm2, cm3, cm4, cm5, cm6 = st.columns(6)
        with cm1:
            st.metric("mAP50", f"{final['metrics/mAP50(B)']:.4f}",
                      delta=f"Best {best_map50:.4f} @E{best_map50_ep}")
        with cm2:
            st.metric("mAP50-95", f"{final['metrics/mAP50-95(B)']:.4f}")
        with cm3:
            st.metric("Precision", f"{final['metrics/precision(B)']:.4f}")
        with cm4:
            st.metric("Recall", f"{final['metrics/recall(B)']:.4f}")
        with cm5:
            st.metric("Train Loss", f"{final['train/box_loss']:.4f}")
        with cm6:
            st.metric("Val Loss", f"{final['val/box_loss']:.4f}")

        st.markdown(
            f'<div style="color:#64748b;font-size:0.8rem;margin-top:4px;">'
            f'总训练时长: {df["time"].iloc[-1]:.0f}s ≈ {df["time"].iloc[-1]/3600:.1f}h · '
            f'设备: NVIDIA GeForce RTX 4060 Laptop · Batch=8 · imgsz=640'
            f'</div>', unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ---- Loss ----
        st.markdown("### 📉 损失曲线 (Loss Convergence)")
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        plt.style.use("dark_background")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#0f1729")
        fig1.patch.set_facecolor("#0f1729")

        ax1.plot(df["epoch"], df["train/box_loss"], color="#818cf8", lw=2, label="Box Loss")
        ax1.plot(df["epoch"], df["train/cls_loss"], color="#c084fc", lw=2, label="Cls Loss")
        ax1.plot(df["epoch"], df["train/dfl_loss"], color="#f59e0b", lw=2, label="DFL Loss")
        ax1.set_title("Training Loss", color="#e2e8f0", fontweight="bold")
        ax1.set_xlabel("Epoch", color="#94a3b8")
        ax1.set_ylabel("Loss", color="#94a3b8")
        ax1.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#c9d1d9")
        ax1.grid(alpha=0.15, color="#ffffff")
        ax1.tick_params(colors="#94a3b8")

        ax2.plot(df["epoch"], df["val/box_loss"], color="#818cf8", lw=2, label="Box Loss")
        ax2.plot(df["epoch"], df["val/cls_loss"], color="#c084fc", lw=2, label="Cls Loss")
        ax2.plot(df["epoch"], df["val/dfl_loss"], color="#f59e0b", lw=2, label="DFL Loss")
        ax2.set_title("Validation Loss", color="#e2e8f0", fontweight="bold")
        ax2.set_xlabel("Epoch", color="#94a3b8")
        ax2.set_ylabel("Loss", color="#94a3b8")
        ax2.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#c9d1d9")
        ax2.grid(alpha=0.15, color="#ffffff")
        ax2.tick_params(colors="#94a3b8")

        fig1.tight_layout()
        st.pyplot(fig1)

        # ---- mAP ----
        st.markdown("### 🎯 检测精度 (Detection Accuracy)")
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax in [ax3, ax4]:
            ax.set_facecolor("#0f1729")
        fig2.patch.set_facecolor("#0f1729")

        ax3.plot(df["epoch"], df["metrics/mAP50(B)"], color="#a78bfa", lw=2.5, marker="o", ms=3)
        ax3.axhline(y=best_map50, color="#ef4444", ls="--", alpha=0.5, lw=1)
        ax3.annotate(f"Best: {best_map50:.4f}", xy=(best_map50_ep, best_map50),
                     xytext=(best_map50_ep + 3, best_map50 - 0.04),
                     color="#ef4444", fontweight="bold", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#ef4444", alpha=0.5))
        ax3.set_title("mAP50 (IoU=0.5)", color="#e2e8f0", fontweight="bold")
        ax3.set_xlabel("Epoch", color="#94a3b8")
        ax3.set_ylabel("mAP50", color="#94a3b8")
        ax3.grid(alpha=0.15, color="#ffffff")
        ax3.tick_params(colors="#94a3b8")

        ax4.plot(df["epoch"], df["metrics/mAP50-95(B)"], color="#34d399", lw=2.5, marker="o", ms=3)
        best_map95 = df["metrics/mAP50-95(B)"].max()
        best_map95_ep = int(df.loc[df["metrics/mAP50-95(B)"].idxmax(), "epoch"])
        ax4.axhline(y=best_map95, color="#ef4444", ls="--", alpha=0.5, lw=1)
        ax4.annotate(f"Best: {best_map95:.4f}", xy=(best_map95_ep, best_map95),
                     xytext=(best_map95_ep + 3, best_map95 - 0.02),
                     color="#ef4444", fontweight="bold", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#ef4444", alpha=0.5))
        ax4.set_title("mAP50-95 (IoU=0.50:0.95)", color="#e2e8f0", fontweight="bold")
        ax4.set_xlabel("Epoch", color="#94a3b8")
        ax4.set_ylabel("mAP50-95", color="#94a3b8")
        ax4.grid(alpha=0.15, color="#ffffff")
        ax4.tick_params(colors="#94a3b8")

        fig2.tight_layout()
        st.pyplot(fig2)

        # ---- P/R ----
        st.markdown("### 📊 精确率与召回率 (Precision & Recall)")
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(13, 4))
        for ax in [ax5, ax6]:
            ax.set_facecolor("#0f1729")
        fig3.patch.set_facecolor("#0f1729")

        ax5.fill_between(df["epoch"], df["metrics/precision(B)"], alpha=0.15, color="#f472b6")
        ax5.plot(df["epoch"], df["metrics/precision(B)"], color="#f472b6", lw=2)
        ax5.set_title("Precision (精确率)", color="#e2e8f0", fontweight="bold")
        ax5.set_xlabel("Epoch", color="#94a3b8")
        ax5.set_ylabel("Precision", color="#94a3b8")
        ax5.grid(alpha=0.15, color="#ffffff")
        ax5.tick_params(colors="#94a3b8")

        ax6.fill_between(df["epoch"], df["metrics/recall(B)"], alpha=0.15, color="#60a5fa")
        ax6.plot(df["epoch"], df["metrics/recall(B)"], color="#60a5fa", lw=2)
        ax6.set_title("Recall (召回率)", color="#e2e8f0", fontweight="bold")
        ax6.set_xlabel("Epoch", color="#94a3b8")
        ax6.set_ylabel("Recall", color="#94a3b8")
        ax6.grid(alpha=0.15, color="#ffffff")
        ax6.tick_params(colors="#94a3b8")

        fig3.tight_layout()
        st.pyplot(fig3)
