#!/usr/bin/env python3
"""
ImageNet 数据集解压和组织脚本
将tar.gz文件解压并组织成PyTorch可用的目录结构
"""

import os
import tarfile
import shutil
from pathlib import Path
import subprocess
import sys

# ImageNet数据集路径
IMAGENET_ROOT = "/nas-ctm01/datasets/public/wenwu/imagenet1k"

def extract_tar_gz(tar_path, extract_to, desc=""):
    """
    解压tar.gz文件
    """
    print(f"\n正在解压 {desc}: {os.path.basename(tar_path)}")
    print(f"目标目录: {extract_to}")
    
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        # 使用tar命令解压（更快）
        cmd = f"tar -xzf {tar_path} -C {extract_to}"
        print(f"执行命令: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {desc} 解压完成")
            return True
        else:
            print(f"✗ 命令行解压失败: {result.stderr}")
            print("尝试使用Python tarfile模块...")
            
            # 备用方案：使用Python的tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                # 获取总文件数
                members = tar.getmembers()
                total = len(members)
                
                # 逐个解压并显示进度
                for i, member in enumerate(members):
                    tar.extract(member, extract_to)
                    if i % 100 == 0:
                        progress = (i / total) * 100
                        sys.stdout.write(f'\r解压进度: {progress:.1f}% ({i}/{total} files)')
                        sys.stdout.flush()
                
                print(f"\n✓ {desc} 解压完成")
                return True
                
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False

def organize_imagenet_structure():
    """
    组织ImageNet目录结构为标准格式
    期望的结构:
    imagenet1k/
    ├── train/
    │   ├── n01440764/
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
    """
    print("\n" + "="*60)
    print("组织ImageNet目录结构")
    print("="*60)
    
    train_tar = os.path.join(IMAGENET_ROOT, "train_images_0.tar.gz")
    val_tar = os.path.join(IMAGENET_ROOT, "val_images.tar.gz")
    
    train_dir = os.path.join(IMAGENET_ROOT, "train")
    val_dir = os.path.join(IMAGENET_ROOT, "val")
    
    # 1. 解压验证集
    if os.path.exists(val_tar):
        if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
            extract_tar_gz(val_tar, IMAGENET_ROOT, "验证集")
            
            # 检查解压后的结构
            if os.path.exists(val_dir):
                print(f"验证集已解压到: {val_dir}")
                
                # 如果val目录下直接是图片，需要组织成子文件夹
                val_files = os.listdir(val_dir)
                if val_files and val_files[0].endswith(('.JPEG', '.jpg', '.png')):
                    print("检测到验证集图片在根目录，需要组织到类别文件夹...")
                    organize_val_images(val_dir)
        else:
            print(f"✓ 验证集已存在: {val_dir}")
    else:
        print(f"⚠️ 验证集压缩包不存在: {val_tar}")
    
    # 2. 解压训练集
    if os.path.exists(train_tar):
        if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
            extract_tar_gz(train_tar, IMAGENET_ROOT, "训练集")
            
            # 训练集通常已经组织好了，检查结构
            if os.path.exists(train_dir):
                print(f"训练集已解压到: {train_dir}")
                
                # 检查是否需要进一步解压每个类别的tar文件
                train_files = os.listdir(train_dir)
                if train_files and train_files[0].endswith('.tar'):
                    print("检测到训练集包含类别tar文件，解压中...")
                    extract_train_class_tars(train_dir)
        else:
            print(f"✓ 训练集已存在: {train_dir}")
    else:
        print(f"⚠️ 训练集压缩包不存在: {train_tar}")
    
    # 3. 统计数据集信息
    print_dataset_stats()

def organize_val_images(val_dir):
    """
    将验证集图片组织到对应的类别文件夹
    需要val_annotations.txt文件来映射图片到类别
    """
    val_annotations = os.path.join(IMAGENET_ROOT, "val_annotations.txt")
    
    if not os.path.exists(val_annotations):
        print("⚠️ 缺少val_annotations.txt文件，尝试使用默认组织方式...")
        # 如果没有标注文件，创建一个临时的单类别文件夹
        temp_class_dir = os.path.join(val_dir, "temp_class")
        os.makedirs(temp_class_dir, exist_ok=True)
        
        for img_file in os.listdir(val_dir):
            if img_file.endswith(('.JPEG', '.jpg', '.png')):
                src = os.path.join(val_dir, img_file)
                dst = os.path.join(temp_class_dir, img_file)
                shutil.move(src, dst)
        
        print(f"✓ 图片已移动到: {temp_class_dir}")
        print("  注意：需要正确的标注文件来组织到对应类别")
        return
    
    # 读取标注文件
    print("读取验证集标注文件...")
    image_to_class = {}
    
    with open(val_annotations, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                image_to_class[img_name] = class_id
    
    # 组织图片
    print("组织验证集图片到类别文件夹...")
    for img_file in os.listdir(val_dir):
        if img_file.endswith(('.JPEG', '.jpg', '.png')):
            if img_file in image_to_class:
                class_id = image_to_class[img_file]
                class_dir = os.path.join(val_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_dir, img_file)
                dst = os.path.join(class_dir, img_file)
                shutil.move(src, dst)
    
    print("✓ 验证集图片组织完成")

def extract_train_class_tars(train_dir):
    """
    解压训练集中每个类别的tar文件
    """
    tar_files = [f for f in os.listdir(train_dir) if f.endswith('.tar')]
    total = len(tar_files)
    
    print(f"找到 {total} 个类别tar文件需要解压")
    
    for i, tar_file in enumerate(tar_files):
        class_name = tar_file.replace('.tar', '')
        class_dir = os.path.join(train_dir, class_name)
        tar_path = os.path.join(train_dir, tar_file)
        
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, exist_ok=True)
            
            # 解压
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(class_dir)
            
            # 删除tar文件以节省空间
            os.remove(tar_path)
            
        if (i + 1) % 10 == 0:
            print(f"进度: {i + 1}/{total} 类别已解压")
    
    print("✓ 所有训练集类别解压完成")

def print_dataset_stats():
    """
    打印数据集统计信息
    """
    print("\n" + "="*60)
    print("ImageNet 数据集统计")
    print("="*60)
    
    train_dir = os.path.join(IMAGENET_ROOT, "train")
    val_dir = os.path.join(IMAGENET_ROOT, "val")
    
    if os.path.exists(train_dir):
        train_classes = [d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
        
        total_train_images = 0
        for class_dir in train_classes:
            class_path = os.path.join(train_dir, class_dir)
            images = [f for f in os.listdir(class_path) 
                     if f.endswith(('.JPEG', '.jpg', '.png'))]
            total_train_images += len(images)
        
        print(f"训练集:")
        print(f"  - 类别数: {len(train_classes)}")
        print(f"  - 图片总数: {total_train_images}")
        print(f"  - 平均每类: {total_train_images // len(train_classes) if train_classes else 0} 张")
    else:
        print("训练集: 未找到")
    
    if os.path.exists(val_dir):
        val_classes = [d for d in os.listdir(val_dir) 
                      if os.path.isdir(os.path.join(val_dir, d))]
        
        total_val_images = 0
        for class_dir in val_classes:
            class_path = os.path.join(val_dir, class_dir)
            images = [f for f in os.listdir(class_path) 
                     if f.endswith(('.JPEG', '.jpg', '.png'))]
            total_val_images += len(images)
        
        print(f"\n验证集:")
        print(f"  - 类别数: {len(val_classes)}")
        print(f"  - 图片总数: {total_val_images}")
        print(f"  - 平均每类: {total_val_images // len(val_classes) if val_classes else 0} 张")
    else:
        print("验证集: 未找到")

def check_and_prepare():
    """
    检查并准备ImageNet数据集
    """
    print("="*60)
    print("ImageNet 数据集准备工具")
    print("="*60)
    
    # 检查压缩包
    files_to_check = [
        ("train_images_0.tar.gz", "24GB", "训练集"),
        ("val_images.tar.gz", "6.5GB", "验证集")
    ]
    
    print("\n检查压缩包...")
    for filename, expected_size, desc in files_to_check:
        filepath = os.path.join(IMAGENET_ROOT, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024**3)  # 转换为GB
            print(f"✓ {desc}: {filename} ({size:.1f}GB)")
        else:
            print(f"✗ {desc}: {filename} 不存在")
    
    # 检查是否已解压
    train_dir = os.path.join(IMAGENET_ROOT, "train")
    val_dir = os.path.join(IMAGENET_ROOT, "val")
    
    print("\n检查解压状态...")
    if os.path.exists(train_dir) and os.listdir(train_dir):
        print(f"✓ 训练集已解压: {train_dir}")
    else:
        print(f"⚠️ 训练集未解压")
    
    if os.path.exists(val_dir) and os.listdir(val_dir):
        print(f"✓ 验证集已解压: {val_dir}")
    else:
        print(f"⚠️ 验证集未解压")
    
    # 询问是否继续
    print("\n" + "="*60)
    print("选择操作:")
    print("1. 解压并组织数据集")
    print("2. 只显示统计信息")
    print("3. 退出")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '1':
        organize_imagenet_structure()
    elif choice == '2':
        print_dataset_stats()
    else:
        print("退出")
        sys.exit(0)

def create_val_annotations():
    """
    创建验证集标注文件（如果需要）
    """
    print("\n创建验证集标注文件...")
    
    # 这里可以下载ILSVRC2012_devkit_t12.tar.gz来获取标注
    # 或者使用预定义的映射
    
    val_annotations_content = """# ImageNet验证集标注文件
# 格式: 图片名 类别ID 
# 这是一个示例，实际使用需要完整的标注文件
# 可以从 https://image-net.org/download.php 下载开发工具包获取
"""
    
    val_annotations_path = os.path.join(IMAGENET_ROOT, "val_annotations.txt")
    
    if not os.path.exists(val_annotations_path):
        with open(val_annotations_path, 'w') as f:
            f.write(val_annotations_content)
        print(f"✓ 创建了标注文件模板: {val_annotations_path}")
        print("  注意：需要下载完整的标注文件才能正确组织验证集")

if __name__ == "__main__":
    check_and_prepare()