# Deep Learning Waste Classification

A Deep Learning project using ResNet50 to classify waste into 12 categories with high accuracy and real-time detection capabilities.

## Features

- **Interactive Menu**: Easy-to-use command-line interface
- **Model Training**: Train ResNet50 model from scratch
- **Unified Web Demo**: Upload images and live camera detection in one interface
- **Real-time Detection**: Live waste classification from webcam
- **Automatic Setup**: Creates necessary folders automatically
- **12-Class Classification**: Supports 12 distinct waste categories
- **High Accuracy**: Achieves ~90% classification accuracy

## Project Structure

```
Project/
├── main.py                 # Main entry point with interactive menu
├── requirements.txt        # Dependencies
├── raw_dataset/           # Raw dataset (place your dataset here)
├── data/                  # Processed dataset (auto-created)
│   ├── train/             # Training data
│   └── test/              # Test data
├── src/                   # Source code
│   ├── model.py           # ResNet50 model implementation
│   ├── train.py           # Training script
│   ├── prepare_data.py    # Data preparation
├── demo/                  # Demo applications
│   └── web.py             # Unified Flask web application
├── templates/             # HTML templates
│   └── index.html         # Web interface with tabs
├── model/                 # Trained models (auto-created)
└── images/                # Screenshots & training plots (auto-created)
```

## Installation and Usage

### 1. Create environment and activate
```bash
python -m venv venv
source ./venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate      # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Place your raw dataset in `raw_dataset/` folder with each class in separate subdirectories:
```
raw_dataset/
├── battery/
├── biological/
├── brown-glass/
├── cardboard/
├── clothes/
├── green-glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
├── trash/
└── white-glass/
```

### 4. Run the System
```bash
python main.py
```

### 5. Interactive Menu Options

**Main Menu:**
1. **Training model** - Train a new ResNet50 model
2. **Demo** - Launch unified web interface
3. **Quit** - Exit the application


## Usage Instructions

### Data Preparation
The system automatically:
- Detects all classes in `raw_dataset/`
- Splits data 80/20 for train/test
- Creates balanced dataset structure
- Supports exactly 12 waste classes
- Balanced dataset with 80/20 train/test split

### Unified Web Demo
Access: `http://127.0.0.1:5000`

**Upload Tab:**
- Upload or drag & drop images
- View classification results with confidence scores
- See top 3 predictions with probabilities

**Live Camera Tab:**
- Click "Start Camera" for real-time detection
- Live waste classification from webcam
- Click "Stop Camera" to end session

## Dataset Classes (12 Categories)

The system classifies waste into 12 distinct categories:

1. **Battery** 🔋 - Electronic batteries and power cells
2. **Biological** 🌿 - Organic waste and food scraps
3. **Brown Glass** 🍺 - Brown/amber glass containers
4. **Cardboard** 📦 - Cardboard boxes and packaging
5. **Clothes** 👕 - Textile materials and clothing
6. **Green Glass** 🍾 - Green glass bottles and containers
7. **Metal** 🥫 - Metal cans, containers, and objects
8. **Paper** 📄 - Paper documents and materials
9. **Plastic** 🥤 - Plastic bottles, containers, and items
10. **Shoes** 👟 - Footwear and shoe materials
11. **Trash** 🗑️ - General non-recyclable waste
12. **White Glass** ⚪ - Clear/white glass containers

## Model Configuration

**Training Parameters:**
- Architecture: ResNet50 (custom implementation)
- Classes: 12 waste categories
- Batch size: 32
- Learning rate: 5e-4
- Max epochs: 50 (with early stopping)
- Early stopping: 8 patience
- Image size: 224x224
- Optimizer: AdamW with weight decay

**Features:**
- Data augmentation
- Class weight balancing
- Learning rate scheduling
- Gradient clipping
- Training progress visualization

## System Requirements

- Python 3.10+
- PyTorch
- OpenCV (for camera functionality)
- Flask (for web interface)
- CUDA (optional, for GPU acceleration)
