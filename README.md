# RoadDamage AI - 基于 YOLOv8 的道路损伤检测系统

本项目基于 RDD2022 多国家道路损伤数据集，使用 YOLOv8 训练道路损伤检测模型，并提供 Streamlit 可视化前端页面。系统支持道路图片上传、批量检测、检测结果可视化、CSV 报告导出、模型性能展示、道路健康评分和养护建议生成。

## 项目目标

课题方向：基于 YOLOv8 的多国家道路损伤检测系统设计与实现。

系统面向以下 4 类道路损伤：

| 类别编号 | 中文名称 | 英文名称 |
|---|---|---|
| D00 | 纵向裂缝 | Longitudinal Crack |
| D10 | 横向裂缝 | Transverse Crack |
| D20 | 龟裂/网状裂缝 | Alligator Crack |
| D40 | 坑洞 | Pothole |

## 技术栈

| 模块 | 技术 |
|---|---|
| 目标检测 | YOLOv8 / Ultralytics |
| 深度学习框架 | PyTorch |
| 前端展示 | Streamlit |
| 图像处理 | OpenCV / Pillow |
| 数据分析 | Pandas / Matplotlib |
| 数据格式 | RDD2022 XML 标注、YOLO txt 标注 |

## 前端页面功能

前端主程序位于 `app/app.py`，通过 Streamlit 运行。

### 1. 检测工作台

- 支持上传单张或多张道路图片。
- 支持使用内置演示图片快速检测。
- 支持切换本地已训练模型权重：
  - `YOLOv8n - 快速检测`
  - `YOLOv8s - 精度优先`
- 支持调节置信度阈值。
- 显示批量检测汇总，包括图片数量、检测目标数、平均健康评分、总耗时。
- 显示每张图片的原图和检测结果图。
- 显示 D00 / D10 / D20 / D40 分类统计和柱状图。
- 支持检测框局部放大，便于观察损伤区域细节。
- 支持下载当前图片 CSV 明细。
- 支持下载批量 CSV 检测报告。

### 2. 道路健康评分

系统会根据不同损伤类别和数量计算 0-100 分的道路健康评分。

- 分数越高表示路面状态越好。
- D40 坑洞、D20 龟裂等更严重病害会产生更高扣分。
- 页面会显示健康等级，如健康、轻微损伤、中度损伤、重度损伤。

### 3. 养护建议

系统会根据检测结果自动生成养护建议，例如：

- 检测到坑洞 D40 时，建议优先现场核查并进行坑槽修补。
- 检测到龟裂 D20 时，建议评估基层承载情况。
- 检测到纵向/横向裂缝时，建议灌缝封闭或纳入周期性养护计划。

### 4. 模型性能仪表盘

页面会读取训练结果 CSV，展示：

- mAP50
- mAP50-95
- Precision
- Recall
- 参数量
- 权重文件大小
- Loss 曲线
- mAP / Precision / Recall 曲线

### 5. 模型对比页面

对比本地 YOLOv8n 与 YOLOv8s 模型：

- 精度指标对比
- 模型体量对比
- 推理特点
- 推荐使用场景

### 6. 历史检测记录

在当前 Streamlit 会话中，系统会自动记录检测历史：

- 批次级历史记录
- 图片级历史记录
- 累计图片数、累计目标数、平均健康评分
- 支持下载批次历史 CSV
- 支持下载图片级历史 CSV
- 支持清空当前会话历史

### 7. 主题与工具栏

- 支持 Streamlit 自带 Light / Dark / System 主题切换。
- 自定义页面样式已适配浅色和深色模式。
- 已隐藏不必要的 Deploy 开发入口。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如需 GPU 推理或训练，请安装与本机 CUDA 匹配的 PyTorch 版本。

### 2. 启动前端页面

```bash
python -m streamlit run app/app.py --server.port 8501 --server.address 127.0.0.1
```

浏览器打开：

```text
http://127.0.0.1:8501
```

## 模型权重

本项目当前使用本地已训练权重：

| 模型 | 路径 | 用途 |
|---|---|---|
| YOLOv8n | `app/weights/yolov8n_best.pt` | 快速检测 |
| YOLOv8s | `app/weights/yolov8s_best.pt` | 精度优先 |

前端页面不会调用官方预训练 `yolov8n.pt` 或 `yolov8s.pt`，下拉框中使用的是项目本地训练好的权重。

## 数据集说明

本项目使用 RDD2022 中 Japan、India、Czech 三个国家的数据。

数据处理流程：

```text
原始 RDD2022 XML 标注
  -> scripts/xml_to_yolo.py 转换为 YOLO txt
  -> scripts/split_dataset.py 按 8:1:1 划分 train/val/test
  -> data.yaml 提供 YOLO 训练配置
```

最终数据配置位于 `data.yaml`：

```yaml
path: datasets/RDD2022_Selected
train: images/train
val: images/val
test: images/test
nc: 4
names: ['D00', 'D10', 'D20', 'D40']
```

注意：完整数据集体积较大，`datasets/`、`Japan/`、`India/`、`Czech/` 已在 `.gitignore` 中排除，不建议直接上传到 GitHub。

## 训练脚本

| 脚本 | 功能 |
|---|---|
| `scripts/xml_to_yolo.py` | 将 XML 标注转换为 YOLO txt |
| `scripts/split_dataset.py` | 生成 train/val/test 数据划分 |
| `scripts/train_yolov8s.py` | 训练 YOLOv8s |
| `scripts/resume_train.py` | YOLOv8n 断点续训 |
| `scripts/resume_yolov8s.py` | YOLOv8s 断点续训 |

训练 YOLOv8s：

```bash
python scripts/train_yolov8s.py
```

## 项目结构

```text
road-detection-main/
├── app/
│   ├── app.py                         # Streamlit 前端主程序
│   ├── weights/                       # 本地训练权重
│   ├── test_images/                   # 演示图片
│   ├── training_results.csv           # YOLOv8n 训练指标
│   └── training_results_yolov8s.csv   # YOLOv8s 训练指标
├── scripts/
│   ├── xml_to_yolo.py
│   ├── split_dataset.py
│   ├── train_yolov8s.py
│   ├── resume_train.py
│   └── resume_yolov8s.py
├── data.yaml
├── requirements.txt
├── README.md
└── 项目目标.md
```

## 已实现功能总结

- YOLOv8 道路损伤检测
- 本地模型权重推理
- 单图/批量图片检测
- 检测结果可视化
- 检测框局部放大
- CSV 报告导出
- 道路健康评分
- 养护建议生成
- 模型性能仪表盘
- 模型对比页面
- 历史检测记录
- Light / Dark / System 主题适配

## 致谢

- Ultralytics YOLOv8
- RDD2022 Road Damage Dataset
- Streamlit
