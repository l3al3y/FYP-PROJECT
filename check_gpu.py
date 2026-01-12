import torch
import sys

def check_system():
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    
    print("\n-------------------------------------------")
    print("🔍 CHECKING FOR GPU...")
    print("-------------------------------------------")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ SUCCESS! GPU DETECTED!")
        print(f"   Name:       {gpu_name}")  # Should say "NVIDIA GeForce RTX 3070 Ti"
        print(f"   Count:      {gpu_count} GPU(s) found")
        print(f"   Index:      {current_device}")
        print(f"   CUDA Ver:   {torch.version.cuda}")
        print("\n🚀 You are ready to train at full speed.")
    else:
        print("❌ FAILURE: No GPU detected.")
        print("   Your code is running on CPU (Slow).")
        print("\n👇 FIX IT:")
        print("Run this command in terminal:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    check_system()