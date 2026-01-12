import os
import zipfile
import requests
import torch
from ultralytics import YOLO

DOWNLOAD_URL = "https://app.roboflow.com/ds/5JtotdiUwa?key=CAjyIC6PHt"
DATASET_DIR = "my_dataset" 

def check_gpu():
   
    print("\n🔍 Checking Hardware...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"SUCCESS: GPU Detected: {gpu_name}")
        print(" Training will be FAST.")
        return 0  # 0 is the ID of the first GPU
    else:
        print("❌ WARNING: GPU NOT DETECTED!")
        print("   The code is trying to run on CPU (Slow).")
        print("   Did you run the 'pip install ... cu121' command?")
        # We return 'cpu' or 0. If you want to force stop, you can exit here.
        return 'cpu'

def download_and_unzip():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    zip_path = os.path.join(DATASET_DIR, "roboflow.zip")

    # Only download if zip doesn't exist to save time
    if not os.path.exists(zip_path) and not os.path.exists(os.path.join(DATASET_DIR, "data.yaml")):
        print(f"⬇️ Downloading dataset...")
        response = requests.get(DOWNLOAD_URL, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print("❌ Download failed.")
            return False

    if not os.path.exists(os.path.join(DATASET_DIR, "data.yaml")):
        print(f"📦 Unzipping...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("✅ Unzip complete.")
    
    return True

def main():
    # 1. Setup Data
    download_and_unzip()

    # 2. Check GPU
    device_to_use = check_gpu()

    # 3. Load Model
    print("\n⬇️ Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt') 

    print("🔥 Starting Training...")
    
    # Get absolute path for config
    data_config_path = os.path.abspath(os.path.join(DATASET_DIR, "data.yaml"))

    results = model.train(
            data=data_config_path,
            epochs=50,
            imgsz=640,
            batch=16,          
            device=device_to_use, # Forces the GPU usage
            plots=True,
            name='smart_checkout_model'
        )
    
        
    print("\n🎉 TRAINING COMPLETE!")

if __name__ == '__main__':
    main()