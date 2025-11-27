# Deep Learning Waste Classification

A Deep Learning project using ResNet50 to classify waste into 6 categories: cardboard, glass, metal, paper, plastic, trash.

## Features

- **Interactive Menu**: Easy-to-use command-line interface
- **Model Training**: Train ResNet50 model from scratch
- **Web Demo**: Upload images via beautiful web interface
- **Camera Demo**: Real-time waste detection from webcam
- **Automatic Setup**: Creates necessary folders automatically

## Project Structure

```
Project/
├── main.py             # Main entry point with interactive menu
├── requirements.txt    # Dependencies
├── archive/            # Raw dataset
├── data/               # Dataset
│   ├── train/          # Training data
│   └── test/           # Test data
├── src/                # Source code
│   ├── model.py        # ResNet50 model implementation
│   ├── train.py        # Training script
│   └── prepare_data.py # Data preparation utilities
├── demo/               # Demo applications
│   ├── web.py          # Flask web application
│   └── camera.py       # Camera demo
├── templates/          # HTML templates
│   └── index.html      # Web interface
├── model/              # Trained models (auto-created)
└── images/             # Screenshots & training plots (auto-created)
```

## Installation and Usage

### 1. Create environment and activate
```bash
python -m venv venv
source ./venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.Run the System
```bash
python main.py
```

### 4. Interactive Menu Options

**Main Menu:**
1. **Training model** - Train a new ResNet50 model
2. **Demo web** - Launch web interface for image upload
3. **Demo camera** - Start real-time camera detection
4. **Quit** - Exit the application

**After Training:**
- Demo web
- Demo camera  
- Back to main menu

**No Model Found:**
- Train model
- Back to main menu

## Usage Instructions

### Web Demo
- Access: `http://127.0.0.1:5000`
- Upload or drag & drop images
- View classification results with confidence scores
- See top 3 predictions

### Camera Demo
- Real-time waste detection
- Press 's' to save screenshot
- Press 'q' to quit
- Screenshots saved to `images/` folder

## Dataset Classes

6 waste categories supported:
- **Cardboard** - Cardboard materials
- **Glass** - Glass containers and bottles
- **Metal** - Metal cans and containers
- **Paper** - Paper materials
- **Plastic** - Plastic containers and bottles
- **Trash** - General waste

## Model Configuration

**Training Parameters:**
- Architecture: ResNet50
- Classes: 6
- Batch size: 32
- Learning rate: 3e-4
- Max epochs: 100
- Early stopping: 15 patience
- Image size: 224x224
- Optimizer: AdamW with weight decay

**Features:**
- Data augmentation
- Class weight balancing
- Learning rate scheduling
- Gradient clipping
- Training progress visualization

## System Requirements

- Python 3.7+
- PyTorch
- OpenCV (for camera demo)
- Flask (for web demo)
- CUDA (optional, for GPU acceleration)
