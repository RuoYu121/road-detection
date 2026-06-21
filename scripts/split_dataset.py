"""
划分 train/val/test 数据集 (8:1:1)
+ 整理演示图片到 app/test_images/
"""
import os
import shutil
import random

random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # roaddetection/
LABEL_DIR = os.path.join(ROOT, "datasets", "temp_all_labels")
OUTPUT_DIR = os.path.join(ROOT, "datasets", "RDD2022_Selected")

# 图片来源
IMG_SOURCES = [
    os.path.join(ROOT, "Japan", "train", "images"),
    os.path.join(ROOT, "India", "train", "images"),
    os.path.join(ROOT, "Czech", "train", "images"),
]

# ---- 创建输出目录 ----
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# ---- 收集所有有标签的图片 ----
all_bases = []
for f in os.listdir(LABEL_DIR):
    if f.endswith(".txt"):
        base = os.path.splitext(f)[0]
        # 确认图片存在
        found = False
        for src_dir in IMG_SOURCES:
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if os.path.exists(os.path.join(src_dir, base + ext)):
                    found = True
                    break
            if found:
                break
        if found:
            all_bases.append(base)

print(f"有效标注图片总数: {len(all_bases)}")

# ---- 8:1:1 随机打乱 ----
random.shuffle(all_bases)
n = len(all_bases)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

train_bases = all_bases[:n_train]
val_bases = all_bases[n_train:n_train + n_val]
test_bases = all_bases[n_train + n_val:]

print(f"训练集 (train): {len(train_bases)}")
print(f"验证集 (val):   {len(val_bases)}")
print(f"测试集 (test):  {len(test_bases)}")


def copy_files(base_list, split_name):
    """复制图片和标签到指定 split 目录"""
    img_count = 0
    for base in base_list:
        # 找图片
        img_copied = False
        for src_dir in IMG_SOURCES:
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                src = os.path.join(src_dir, base + ext)
                if os.path.exists(src):
                    dst_ext = ext if ext.lower() in [".jpg", ".jpeg", ".png"] else ".jpg"
                    dst = os.path.join(OUTPUT_DIR, "images", split_name, base + dst_ext)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    img_copied = True
                    img_count += 1
                    break
            if img_copied:
                break

        # 复制标签
        label_src = os.path.join(LABEL_DIR, base + ".txt")
        label_dst = os.path.join(OUTPUT_DIR, "labels", split_name, base + ".txt")
        if os.path.exists(label_src) and not os.path.exists(label_dst):
            shutil.copy2(label_src, label_dst)

    return img_count


print("\n复制文件中...")
t1 = copy_files(train_bases, "train")
print(f"  train: {t1} 张图片")
t2 = copy_files(val_bases, "val")
print(f"  val:   {t2} 张图片")
t3 = copy_files(test_bases, "test")
print(f"  test:  {t3} 张图片")

# ---- 整理演示图片 (test 无标注图片) ----
print("\n整理演示图片...")
demo_dir = os.path.join(ROOT, "app", "test_images")
os.makedirs(demo_dir, exist_ok=True)

demo_sources = [
    os.path.join(ROOT, "Japan", "test", "images"),
    os.path.join(ROOT, "India", "test", "images"),
    os.path.join(ROOT, "Czech", "test", "images"),
]

demo_count = 0
for src_dir in demo_sources:
    if not os.path.exists(src_dir):
        continue
    for f in os.listdir(src_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            dst = os.path.join(demo_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src_dir, f), dst)
                demo_count += 1
                if demo_count >= 30:  # 最多取 30 张演示图
                    break
    if demo_count >= 30:
        break

print(f"  演示图片: {demo_count} 张 → app/test_images/")
print("\n数据划分完成!")
