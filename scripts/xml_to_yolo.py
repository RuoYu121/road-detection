"""
XML 标注 → YOLO txt 格式转换
处理 Japan、India、Czech 三个国家的数据
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image

# ---- 配置 ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # roaddetection/
COUNTRIES = ["Japan", "India", "Czech"]

CLASS_MAP = {
    "D00": 0,   # 纵向裂缝
    "D10": 1,   # 横向裂缝
    "D20": 2,   # 龟裂 / 网状裂缝
    "D40": 3,   # 坑洞
}

OUTPUT_DIR = os.path.join(ROOT, "datasets", "temp_all_labels")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 统计 ----
stats = {c: {"total": 0, "converted": 0, "skipped_no_xml": 0, "skipped_no_image": 0} for c in COUNTRIES}

def xml_to_yolo_lines(xml_path, img_w, img_h):
    """解析一个 XML，返回 YOLO 格式的行列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        if xmax <= xmin or ymax <= ymin:
            continue

        # 归一化
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width    = (xmax - xmin) / img_w
        height   = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def process_country(country):
    """处理一个国家"""
    img_dir = os.path.join(ROOT, country, "train", "images")
    xml_dir = os.path.join(ROOT, country, "train", "annotations", "xmls")

    if not os.path.exists(img_dir):
        print(f"  [WARN] {img_dir} 不存在")
        return
    if not os.path.exists(xml_dir):
        print(f"  [WARN] {xml_dir} 不存在")
        return

    # 收集所有有 XML 的图片
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]

    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(xml_dir, xml_file)

        # 找对应的图片文件
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            stats[country]["skipped_no_image"] += 1
            continue

        # 读取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"  [ERROR] 无法读取图片 {img_path}: {e}")
            continue

        # 转换
        lines = xml_to_yolo_lines(xml_path, img_w, img_h)
        if not lines:
            continue  # 没有有效标注

        # 写入 txt
        txt_path = os.path.join(OUTPUT_DIR, base_name + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        stats[country]["converted"] += 1

    stats[country]["total"] = len(xml_files)


# ---- 主流程 ----
print("=" * 60)
print("  RDD2022 XML → YOLO 格式转换")
print("=" * 60)

for country in COUNTRIES:
    print(f"\n处理 {country} ...")
    process_country(country)

# ---- 总结 ----
print("\n" + "=" * 60)
print("  转换结果统计")
print("=" * 60)
grand_total_xml = 0
grand_converted = 0
for country in COUNTRIES:
    s = stats[country]
    grand_total_xml += s["total"]
    grand_converted += s["converted"]
    print(f"  {country}: XML={s['total']}, 成功={s['converted']}, 缺图片={s['skipped_no_image']}")

print(f"\n  总计: 转换 {grand_converted} / {grand_total_xml} 个标注文件")
print(f"  输出目录: {OUTPUT_DIR}")
print("  完成!")
