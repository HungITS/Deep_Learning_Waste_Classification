# 🗑️ Hệ thống nhận diện rác thải

Dự án Deep Learning sử dụng ResNet50 để phân loại rác thải thành 6 loại: cardboard, glass, metal, paper, plastic, trash.

## 📋 Tính năng

- ✅ **Training tự động**: Tự động training model nếu chưa có
- 🌐 **Web Demo**: Upload ảnh qua giao diện web đẹp
- 📷 **Camera Demo**: Nhận diện real-time từ webcam
- 🚀 **Portable**: Dễ dàng chuyển sang máy khác

## 🏗️ Cấu trúc project

```
Project/
├── main.py              # File chính để chạy hệ thống
├── requirements.txt     # Dependencies
├── data/               # Dataset
│   ├── train/          # Dữ liệu training
│   └── test/           # Dữ liệu test
├── src/                # Source code
│   ├── model.py        # ResNet50 model
│   ├── train.py        # Training script
│   └── inference.py    # Inference utilities
├── demo/               # Demo applications
│   ├── web.py          # Flask web application
│   └── camera.py       # Camera demo real-time
├── templates/          # HTML templates
│   └── index.html      # Web interface
├── model/              # Trained models (tự động tạo)
└── images/             # Screenshots & plots (tự động tạo)
```

## 🚀 Cài đặt và chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy hệ thống

**Web Demo (mặc định):**
```bash
python main.py --mode web
```

**Camera Demo:**
```bash
python main.py --mode camera
```

**Bắt buộc training lại:**
```bash
python main.py --force-train --mode web
```

### 3. Sử dụng

**Web Demo:**
- Mở trình duyệt: `http://127.0.0.1:5000`
- Upload hoặc kéo thả ảnh vào giao diện
- Xem kết quả phân loại với độ tin cậy

**Camera Demo:**
- Nhấn `s` để chụp màn hình
- Nhấn `q` để thoát

## 🎯 Quy trình hoạt động

1. **Kiểm tra model**: Tự động kiểm tra folder `model/`
2. **Training**: Nếu không có model → tự động training
3. **Lưu model**: Model được lưu với timestamp
4. **Demo**: Khởi động web hoặc camera demo

## 📊 Dataset

6 loại rác được hỗ trợ:
- 📦 **Cardboard** (Bìa carton)
- 🍶 **Glass** (Thủy tinh) 
- 🥫 **Metal** (Kim loại)
- 📄 **Paper** (Giấy)
- 🥤 **Plastic** (Nhựa)
- 🗑️ **Trash** (Rác thải khác)

## 🔧 Cấu hình

**Training parameters** (trong `src/train.py`):
- Batch size: 32
- Learning rate: 3e-4
- Epochs: 100 (với early stopping)
- Image size: 224x224

**Model**: ResNet50 với 6 classes output
