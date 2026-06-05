# 🛣️ RoadDamage AI — 基于 YOLOv8 的道路损伤检测系统

基于 RDD2022 多国道路损伤数据集，使用 YOLOv8 目标检测算法训练的道路损伤智能识别系统，支持 **4 类道路损伤** 自动检测，并提供 Streamlit 可视化交互界面。

---

## 🚀 新手 3 分钟跑起来

> 写给组员：按下面步骤做，3 分钟就能在自己电脑上跑起来。

### 第一步：拉代码

```bash
git clone https://github.com/RuoYu121/road-detection.git
cd roaddetection
```

### 第二步：装依赖

```bash
pip install -r requirements.txt
```

### 第三步：下载数据集和模型

| 文件 | 获取方式 | 放哪里 | 大小 |
|------|---------|--------|------|
| 数据集 | 📱 微信群下载 3 个 zip (Japan/India/Czech) | 解压到项目根目录 | 1.9 GB |
| 模型权重 | ✅ 已包含在仓库中 | `app/weights/` | 33 MB |
| 演示图片 | ✅ 已包含在仓库中 | `app/test_images/` | 2.3 MB |

> 解压后目录结构应为：`roaddetection/Japan/` `roaddetection/India/` `roaddetection/Czech/`

### 第四步：启动

```bash
streamlit run app/app.py
```

浏览器自动打开 → 上传图片 → 看结果 🎉

---

## 📋 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [模型训练](#模型训练)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [模型对比](#模型对比)
- [常见问题](#常见问题)

---

## 项目简介

| 项目 | 说明 |
|------|------|
| **课题** | 基于 YOLOv8 的多国家道路损伤检测系统设计与实现 |
| **算法** | YOLOv8 (Ultralytics) |
| **框架** | PyTorch 2.6 + CUDA 12.4 |
| **界面** | Streamlit |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| **数据集** | RDD2022 (Japan / India / Czech) |

### 检测类别

| 编号 | 名称 | 说明 |
|:--:|------|------|
| D00 | 纵向裂缝 | Longitudinal Crack |
| D10 | 横向裂缝 | Transverse Crack |
| D20 | 龟裂/网状裂缝 | Alligator Crack |
| D40 | 坑洞 | Pothole |

---

## 环境要求

- **Python**: 3.10+
- **CUDA**: 12.4（GPU 训练必需，CPU 推理也可运行）
- **显存**: 训练建议 8GB+，推理 2GB+ 即可

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：`torch>=2.6` `ultralytics>=8.4` `streamlit>=1.58` `opencv-python` `matplotlib` `pandas`

---

## 快速开始

### 1. 启动检测系统

```bash
cd roaddetection
streamlit run app/app.py
```

浏览器访问 `http://localhost:8501`

### 2. 使用方式

1. 上传路面图片（JPG / PNG）或点击「随机演示」
2. 左侧选择模型（YOLOv8n 更快 / YOLOv8s 更准）
3. 调节置信度阈值滑块
4. 查看检测结果、统计图表，可导出 CSV 报告

---

## 数据集

### 数据来源

RDD2022 数据集，选取 Japan、India、Czech 三个国家的道路图像。

### 数据处理流程

```
原始数据 (XML标注)
  └─ xml_to_yolo.py      → 转为 YOLO txt 格式
  └─ split_dataset.py     → 8:1:1 划分 train/val/test
  └─ 最终数据集: train 9756 / val 1219 / test 1220
```

### 数据集结构

```
datasets/RDD2022_Selected/
├── images/
│   ├── train/    (9756 张)
│   ├── val/      (1219 张)
│   └── test/     (1220 张)
└── labels/
    ├── train/    (9756 个)
    ├── val/      (1219 个)
    └── test/     (1220 个)
```

---

## 模型训练

### 训练 YOLOv8n

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, batch=8, imgsz=640, name="yolov8n_rdd")
```

### 训练 YOLOv8s

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="data.yaml", epochs=50, batch=8, imgsz=640, name="yolov8s_rdd")
```

### 续训（从断点恢复）

```python
model = YOLO("runs/detect/yolov8s_rdd/weights/last.pt")
model.train(resume=True)
```

> ⚠️ 训练前请确保 `data.yaml` 路径正确，数据集已准备就绪。

---

## 使用说明

### 应用界面

| 功能 | 说明 |
|------|------|
| 📤 图片上传 | 支持 JPG / JPEG / PNG |
| 🎲 随机演示 | 从 30 张测试图片中随机选取 |
| 🧠 模型选择 | YOLOv8n（快） / YOLOv8s（准） |
| 🎚️ 置信度滑块 | 0.05~0.95，越低检出越多 |
| 📊 统计图表 | 并排对比 + 损伤统计表 + 柱状图 |
| 📥 导出报告 | CSV 格式，Excel 可直接打开 |
| 📈 训练分析 | Loss/mAP/PR 曲线，最佳值标注 |

### 运行脚本说明

| 脚本 | 功能 |
|------|------|
| `scripts/xml_to_yolo.py` | XML 标注 → YOLO txt 转换 |
| `scripts/split_dataset.py` | 数据集 8:1:1 划分 |
| `scripts/train_yolov8s.py` | YOLOv8s 训练脚本 |
| `scripts/resume_train.py` | YOLOv8n 续训脚本 |
| `scripts/resume_yolov8s.py` | YOLOv8s 续训脚本 |

---

## 项目结构

```
roaddetection/
├── data.yaml                      # 数据集配置文件
├── requirements.txt               # Python 依赖
├── README.md                      # 项目说明
│
├── app/                           # Streamlit 应用
│   ├── app.py                     # 主程序
│   ├── weights/                   # 模型权重
│   │   ├── yolov8n_best.pt        # YOLOv8n 训练权重 (mAP50=0.537)
│   │   └── yolov8s_best.pt        # YOLOv8s 训练权重 (mAP50=0.563)
│   └── test_images/               # 演示图片 (30张)
│
├── datasets/                      # 数据集
│   └── RDD2022_Selected/
│       ├── images/ {train,val,test}
│       └── labels/ {train,val,test}
│
├── scripts/                       # 工具脚本
│   ├── xml_to_yolo.py
│   ├── split_dataset.py
│   ├── train_yolov8s.py
│   ├── resume_train.py
│   └── resume_yolov8s.py
│
├── runs/                          # 训练输出
│   └── detect/
│       ├── yolov8n_rdd-2/         # YOLOv8n 训练结果
│       └── yolov8s_rdd/           # YOLOv8s 训练结果
│
├── Japan/ India/ Czech/           # 原始数据 (RDD2022)
└── raw_data/                      # 额外原始数据
```

---

## 模型对比

| 指标 | YOLOv8n | YOLOv8s |
|------|:--:|:--:|
| 参数量 | 3.0M | 11.1M |
| 显存占用 | ~1.3GB | ~2.5GB |
| 推理速度 | ~7.5 it/s | ~4.5 it/s |
| **mAP50** | **0.537** | **0.563** |
| mAP50-95 | 0.249 | 0.261 |
| Precision | 0.578 | 0.594 |
| Recall | 0.525 | 0.532 |

> 训练设备: RTX 4060 Laptop 8GB | Epochs: 50 | Batch: 8 | imgsz: 640

---

## 常见问题

**Q: 没有 GPU 能运行吗？**
A: 可以。模型推理支持 CPU，修改代码中 `device='cpu'` 即可，但速度会慢一些。

**Q: 训练中断了怎么办？**
A: 训练会保存 `last.pt`，使用 `model.train(resume=True)` 从断点续训。

**Q: 如何添加新的损伤类别？**
A: 修改 `data.yaml` 中的 `nc` 和 `names`，重新处理标注并训练。

**Q: 检测结果不准怎么办？**
A: 尝试降低置信度阈值（0.15~0.2），或使用更准的 YOLOv8s 模型。

**Q: 训练好的模型在哪里？**
A: `app/weights/` 目录下，`yolov8n_best.pt` 和 `yolov8s_best.pt`。

---

## 团队成员

| 角色 | 姓名 | GitHub |
|------|------|--------|
| 组长 | — | [RuoYu121](https://github.com/RuoYu121) |
| 组员 | 待补充 | — |
| 组员 | 待补充 | — |

> 把组员名字和 GitHub 填上去就行

---

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [RDD2022 数据集](https://github.com/sekilab/RoadDamageDetector/)
- [Streamlit](https://streamlit.io/)
