import os
import sys
import subprocess
from pathlib import Path

def check_model_exists():
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_files = list(model_dir.glob("*.pth"))
    return len(model_files) > 0, model_files

def train_model():
    print("=" * 50)
    print("Starting model training...")
    print("=" * 50)
    
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
        return True
    else:
        print("Model not found after training!")
        return False

def get_latest_model():
    model_dir = Path("model")
    model_files = list(model_dir.glob("*.pth"))
    
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getctime)
    return str(latest_model)

def run_web_demo():
    model_path = get_latest_model()
    print(f"Starting web demo with model: {model_path}")
    os.environ['MODEL_PATH'] = os.path.abspath(model_path)
    os.system("cd demo && python web.py")

def run_camera_demo():
    model_path = get_latest_model()
    print(f"Starting camera demo with model: {model_path}")
    os.environ['MODEL_PATH'] = os.path.abspath(model_path)
    os.system("cd demo && python camera.py")

def post_training_menu():
    while True:
        print("\n" + "=" * 40)
        print("Training completed! Choose an option:")
        print("1. Demo web")
        print("2. Demo camera") 
        print("3. Back to main menu")
        print("=" * 40)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            run_web_demo()
        elif choice == "2":
            run_camera_demo()
        elif choice == "3":
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

def no_model_menu():
    while True:
        print("\n" + "=" * 40)
        print("No model found! Choose an option:")
        print("1. Train model")
        print("2. Back to main menu")
        print("=" * 40)
        
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == "1":
            if train_model():
                post_training_menu()
            break
        elif choice == "2":
            break
        else:
            print("Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("WASTE CLASSIFICATION SYSTEM")
        print("=" * 50)
        print("1. Training model")
        print("2. Demo web")
        print("3. Demo camera")
        print("4. Quit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            if train_model():
                post_training_menu()
        elif choice == "2":
            has_model, _ = check_model_exists()
            if has_model:
                run_web_demo()
            else:
                no_model_menu()
        elif choice == "3":
            has_model, _ = check_model_exists()
            if has_model:
                run_camera_demo()
            else:
                no_model_menu()
        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")

