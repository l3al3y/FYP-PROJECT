# Hybrid Self-Checkout System 🛒📷

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logoColor=black" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL" />
  <img src="https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=for-the-badge&logo=Raspberry-Pi&logoColor=white" alt="Raspberry Pi" />
</div>

> **Final Year Project (FYP) submitted for the Bachelor of Computer Engineering degree.**

A smart self-checkout assistant designed to prevent fraud and enhance user experience by cross-referencing barcode scans with real-time computer vision object detection.

## 🎯 Problem Statement

Traditional self-checkout systems rely solely on barcode scanning, making them vulnerable to item switching, missed scans, and deliberate fraud. This project solves that by adding a computer vision layer that visually verifies items.

## 🚀 Overview

This project implements a hybrid validation system that uses **YOLOv8** to visually identify products on the checkout counter and compares them against scanned barcodes. If an item is detected by the camera but not scanned (or vice-versa), the system flags a potential anomaly.

Key features:
- **Real-time Object Detection:** Powered by YOLOv8, trained on a custom dataset of local products (Maggi, Gardenia Bread, Tissues, Toothpaste, etc.).
- **Fraud Prevention:** Logic to detect "missed scans" or "fake scans" by matching visual counts with scanned counts.
- **Occlusion Handling:** Experiments with split-view processing to see items from multiple angles.

## 🏗️ Architecture

Camera → YOLOv8 Detection → Cross-Verification Engine → Barcode Scanner Input → MySQL Database → Approval/Alert

## 🔧 Hardware Requirements

- Raspberry Pi 5
- USB Camera
- USB Barcode Scanner
- HDMI Display

## 📂 Project Structure

```text
├── alerts/                     # System alerts and logs
├── library/                    # Project-specific library dependencies
├── my_dataset/                 # Custom YOLOv8 dataset (Train/Test/Valid)
├── related_papers/             # Research and references
├── runs/                       # YOLOv8 training runs and weights
├── advanced_dashboard.py       # Main dashboard interface (if applicable)
├── check_gpu.py                # Utility to verify GPU availability for PyTorch
├── smart_checkout_assistant.py # Core logic for the checkout assistant
├── splitframes.py              # Utility for video frame processing
├── train_system.py             # Script to train the YOLOv8 model
└── visualize_split_view.py     # Tool to visualize split-view detection
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended for training)
- Ultralytics YOLOv8
- OpenCV
- Pandas/Numpy

## 🏁 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/l3al3y/FYP-PROJECT.git
    cd FYP-PROJECT
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Assistant:**
    ```bash
    python smart_checkout_assistant.py
    ```

4.  **Train the Model (Optional):**
    If you want to retrain on the `my_dataset` folder:
    ```bash
    python train_system.py
    ```

## 📸 Demo

Screenshots and demo videos coming soon

## 📊 Model Performance

- **Current Epochs:** 50
- **Precision:** ~77.4%
- **Recall:** ~72.0%
- **Focus:** The current model favors high precision. Future improvements target Recall to reduce false negatives (missed items).

## 📝 Dataset

The project uses a custom dataset (`my_dataset`) structured for YOLOv8, containing images of:
- Maggi
- Roti (Bread)
- Tisu (Tissues)
- Ubat Gigi (Toothpaste)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

Muhammad Irfan Fahmi, Computer Engineering (Hons) UTeM, CCNA, GitHub: l3al3y, Portfolio: l3al3y.github.io/Portfolio
