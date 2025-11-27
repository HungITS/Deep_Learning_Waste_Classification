import os
import sys
import glob
import argparse
from pathlib import Path

def check_model_exists():
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_files = list(model_dir.glob("*.pth"))
    print(f"Checking model folder: {'Found' if model_files else 'No'} model")
    if model_files:
        print(f"   Found {len(model_files)} model(s): {[f.name for f in model_files]}")
    return len(model_files) > 0, model_files

def train_model():
    print("=" * 50)
    
    import subprocess
    result = subprocess.run([sys.executable, "-m", "src.train"], 
                          capture_output=False, text=True)
    
    print("=" * 50)
    
    if os.path.exists("best_resnet50.pth"):
        import shutil
        import time
        timestamp = int(time.time())
        new_model_path = f"model/resnet50_{timestamp}.pth"
        shutil.move("best_resnet50.pth", new_model_path)
        print(f"Model saved to: {new_model_path}")
        return new_model_path
    else:
        print("Model not found after training!")
        return None

def get_latest_model():
    model_dir = Path("model")
    model_files = list(model_dir.glob("*.pth"))
    
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getctime)
    return str(latest_model)

def run_web_demo(model_path):
    print(f"Starting web demo with model: {model_path}")
    os.environ['MODEL_PATH'] = os.path.abspath(model_path)
    os.system("cd demo && python web.py")

def run_camera_demo(model_path):
    print(f"Starting camera demo with model: {model_path}")
    Path("images").mkdir(exist_ok=True)
    os.environ['MODEL_PATH'] = os.path.abspath(model_path)
    os.system("cd demo && python camera.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Classification System")
    parser.add_argument("--mode", choices=["web", "camera"], default="web")
    args = parser.parse_args()
    
    print("=" * 40)
    
    has_model, existing_models = check_model_exists()
    
    if not has_model:
        print("No model found - Starting training...")
        model_path = train_model()
        if not model_path:
            print("Training failed!")
            sys.exit(1)
    else:
        model_path = get_latest_model()
        print(f"Using existing model: {model_path}")
    
    print("\n" + "=" * 40)
    
    if args.mode == "web":
        run_web_demo(model_path)
    else:
        run_camera_demo(model_path)


