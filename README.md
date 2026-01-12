# Hybrid Self-Checkout System 🛒📷

> **Final Year Project (FYP) submitted for the Bachelor of Computer Engineering degree.**

A smart self-checkout assistant designed to prevent fraud and enhance user experience by cross-referencing barcode scans with real-time computer vision object detection.

## 🚀 Overview

This project implements a hybrid validation system that uses **YOLOv8** to visually identify products on the checkout counter and compares them against scanned barcodes. If an item is detected by the camera but not scanned (or vice-versa), the system flags a potential anomaly.

Key features:
- **Real-time Object Detection:** Powered by YOLOv8, trained on a custom dataset of local products (Maggi, Gardenia Bread, Tissues, Toothpaste, etc.).
- **Fraud Prevention:** Logic to detect "missed scans" or "fake scans" by matching visual counts with scanned counts.
- **Occlusion Handling:** Experiments with split-view processing to see items from multiple angles.

## 📂 Project Structure

```
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
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies:**
    (Ensure you have the necessary libraries installed. A `requirements.txt` generation is recommended if not present).
    ```bash
    pip install ultralytics opencv-python pandas numpy
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
