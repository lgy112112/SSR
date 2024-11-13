# 🌟 SSR: Sample Selection and Review for Segmentation Tasks

Welcome to **SSR**! SSR is a PyTorch-based deep learning framework designed to improve segmentation model performance by implementing a **Sample Selection and Review** process. It identifies **low-confidence samples** during training, logs them, and then revisits them with additional training epochs and augmentations to enhance model robustness and accuracy. 🚀

## 2024/11/13

Init: Didn't get any improvement from experiments. Hopefully we may get it better if some features are introduced in review_epoch



---

## 📋 Features

- **Selective Sample Review**: Dynamically identifies and reviews low-confidence samples based on segmentation quality metrics.
- **Modular Design**: Easily integrates with existing segmentation pipelines and supports **custom augmentations**.
- **Flexible Training and Review**: Configurable training epochs, confidence thresholds, and review cycles.
- **MLflow Integration**: Automatically logs metrics, parameters, and models for easy experiment tracking and visualization.

---

## 📁 Directory Structure

```
SSR/
├── dataset.py            # Custom dataset classes for regular and review samples
├── pipeline.py           # Core training, validation, and review epochs
├── data_get.ipynb        # Get data from kaggle and preprocess
├── train.ipynb           # Training process including benchmarking and review experiment
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/SSR.git
cd SSR
```

### 2️⃣ Install Requirements
Install the necessary packages with:
```bash
pip install kaggle mlflow
```

### 3️⃣ Run a Benchmark Experiment & 4️⃣ Run a Review Experiment
To run a standard benchmark without review epochs, execute:
```bash
data_get.ipynb
train.ipynb
```

### 5️⃣ Experiment Tracking with MLflow
SSR supports **MLflow** logging for easy experiment tracking. Start the MLflow server with:
```bash
mlflow ui
```
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to visualize your experiment metrics and models! 📊✨

---

## ⚙️ Configuration

### Custom Augmentations

Configure your data augmentation transformations:
```python
review_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

---

## 📈 Results and Visualization

SSR logs all metrics, including:
- **Training and Validation Loss**
- **Dice Scores**
- **Review Epoch Performance**

These metrics are visualized using **MLflow** and can be accessed through the MLflow UI. SSR also logs the final model for each experiment, allowing you to compare baseline and review-enhanced results side-by-side. 🔍

---

## 🛠 Future Enhancements

- **Add Support for Additional Metrics**: Include more segmentation quality metrics (e.g., IoU).
- **Flexible Review Strategy**: Enable different strategies for identifying review samples.
- **Enhanced Augmentations**: Experiment with 3D and GAN-based augmentations.

---

## 🤝 Contributions

Contributions are welcome! Feel free to submit pull requests, suggest features, or open issues to help improve SSR. 🙌

---

## Buy me a Coffee please 🐱☕

