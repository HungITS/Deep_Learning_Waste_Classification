import os
import shutil
import random
from pathlib import Path

RAW_DATASET_DIR = "./raw_dataset"
OUTPUT_DIR = "./data"
TRAIN_RATIO = 0.8  # 80% train, 20% test

def prepare_new_dataset():
    # Xóa data cũ
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Tạo thư mục mới
    train_dir = Path(OUTPUT_DIR) / "train"
    test_dir = Path(OUTPUT_DIR) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách tất cả classes
    classes = [d for d in os.listdir(RAW_DATASET_DIR) 
              if os.path.isdir(os.path.join(RAW_DATASET_DIR, d))]
    
    print(f"Found {len(classes)} classes: {classes}")
    
    total_train = 0
    total_test = 0
    
    for class_name in classes:
        class_path = os.path.join(RAW_DATASET_DIR, class_name)
        
        # Lấy tất cả ảnh trong class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle để random
        random.shuffle(images)
        
        # Chia train/test
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Tạo thư mục cho class
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # Copy ảnh train
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = train_class_dir / img
            shutil.copy2(src, dst)
        
        # Copy ảnh test
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = test_class_dir / img
            shutil.copy2(src, dst)
        
        total_train += len(train_images)
        total_test += len(test_images)
        
        print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")
    
    print(f"\nDataset preparation completed:")
    print(f"  Total classes: {len(classes)}")
    print(f"  Train images: {total_train}")
    print(f"  Test images: {total_test}")
    print(f"  Total images: {total_train + total_test}")
    
    return len(classes)

if __name__ == "__main__":
    random.seed(42)  # Reproducible split
    num_classes = prepare_new_dataset()