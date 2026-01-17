# 🎭 Deepfake Image Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A comprehensive comparative study of deep learning architectures for automated deepfake image detection**

![Deepfake Detection Banner](https://via.placeholder.com/1200x300/667eea/ffffff?text=Deepfake+Detection+System)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architectures](#model-architectures)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

This project implements and evaluates three distinct deep learning architectures for binary classification of real vs. fake facial images:

- **EfficientNetB0**: Transfer learning with compound scaling
- **Xception**: Depthwise separable convolutions with extreme Inception
- **MesoNet**: Lightweight specialized architecture for mesoscopic artifact detection

Our research demonstrates that **EfficientNetB0 achieves superior performance with 96.38% accuracy and 0.9911 AUC**, outperforming Xception (94.33% accuracy) and MesoNet (91.78% accuracy) while maintaining computational efficiency with only 5.3 million parameters.

### 🔬 Research Highlights

✅ Comprehensive evaluation on **FaceForensics++ (c40)** dataset  
✅ **160,000 images** (80K real, 80K fake) across 6 manipulation techniques  
✅ Cross-domain testing reveals **dataset diversity > augmentation**  
✅ Dual-metric training strategy (Loss + AUC) for optimal performance  
✅ Complete implementation with reproducible results  

---

## 🌟 Key Features

### Model Comparison

| Model | Parameters | Accuracy | AUC | F1-Score | Computational Cost |
|-------|-----------|----------|-----|----------|-------------------|
| **EfficientNetB0** | 5.3M | **96.38%** | **0.9911** | **0.9637** | Low |
| **Xception** | 22M | 94.33% | 0.9846 | 0.9440 | High |
| **MesoNet** | 62K | 91.78% | 0.9761 | 0.9171 | Very Low |

### Manipulation Techniques Detected

1. **Deepfakes** - Autoencoder-based face swapping
2. **FaceSwap** - Graphics-based identity swap
3. **FaceShifter** - High-fidelity neural face swapping
4. **Face2Face** - Facial reenactment (expression transfer)
5. **NeuralTextures** - GAN-based lip-syncing
6. **DeepFakeDetection (DFD)** - Google/Jigsaw dataset

---

## 🏗️ Model Architectures

### 1. EfficientNetB0 (Recommended)

```
Input (224×224×3)
    ↓
Conv3×3 (32 channels)
    ↓
9× MBConv Blocks with SE
    ↓
Conv1×1 + Global Avg Pool
    ↓
Dropout(0.2) + Dense(1)
    ↓
Sigmoid Output
```

**Key Features:**
- Neural Architecture Search optimized
- Mobile Inverted Bottleneck Convolution (MBConv)
- Squeeze-and-Excitation attention
- Compound scaling method
- **Only 5.3M parameters, 0.39B FLOPs**

### 2. Xception

```
Entry Flow (3 blocks)
    ↓
Middle Flow (8× repetitions)
    ↓
Exit Flow (2 blocks)
    ↓
Dense(512) → Dropout(0.2)
    ↓
Dense(256) → Dropout(0.2)
    ↓
Dense(1, sigmoid)
```

**Key Features:**
- Depthwise separable convolutions
- 36 layers with residual connections
- 23M parameters for rich representations
- Optimal for high-accuracy scenarios

### 3. MesoNet

```
Conv(8, 3×3) → BN → MaxPool
    ↓
Conv(8, 5×5) → BN → MaxPool
    ↓
Conv(16, 5×5) → BN → MaxPool
    ↓
Conv(16, 5×5) → BN → MaxPool
    ↓
Flatten → Dropout(0.5)
    ↓
Dense(16) → Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

**Key Features:**
- Mesoscopic-level artifact detection
- Only 62K parameters
- Trained from scratch
- Ideal for edge deployment

---

## 📊 Dataset

### FaceForensics++ (c40 Compression)

- **Total Images**: 160,000 (perfectly balanced)
  - Real: 80,000 images
  - Fake: 80,000 images
- **Split**: 80% Train / 10% Val / 10% Test
- **Resolution**: 224×224 pixels
- **Preprocessing**: MTCNN face extraction

### Directory Structure

```
FINAL_DATASET/
├── Normal_Dataset/
│   ├── real/
│   │   ├── original_sequences/
│   │   │   ├── youtube/
│   │   │   └── actors/
│   └── fake/
│       ├── Deepfakes/
│       ├── FaceSwap/
│       ├── FaceShifter/
│       ├── Face2Face/
│       ├── NeuralTextures/
│       └── DeepFakeDetection/
└── Augmented_Dataset/
    ├── real/
    └── fake/
```

---

## 🏆 Results

### Within-Domain Performance (Normal → Normal)

| Model | Accuracy | AUC | F1-Score | Loss |
|-------|----------|-----|----------|------|
| **EfficientNetB0** | **96.38%** | **0.9911** | **0.9637** | **0.1187** |
| Xception | 94.33% | 0.9846 | 0.9440 | 0.1672 |
| MesoNet | 91.78% | 0.9761 | 0.9171 | 0.1994 |

### Cross-Domain Performance

#### Robustness Check (Normal → Augmented)

| Model | Accuracy | AUC | F1-Score |
|-------|----------|-----|----------|
| EfficientNetB0 | 93.54% | 0.9775 | 0.9343 |
| Xception | 91.46% | 0.9712 | 0.9118 |
| MesoNet | 82.97% | 0.8996 | 0.8239 |

### Key Insights

1. **EfficientNetB0 is the best overall**: Highest accuracy, AUC, and F1-score
2. **Dataset diversity matters more than augmentation**: Models trained on diverse data generalize better
3. **Transfer learning outperforms from-scratch training**: Pre-trained models converge faster and achieve higher accuracy
4. **Computational efficiency**: EfficientNetB0 achieves best results with 4× fewer parameters than Xception

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Clone Repository

```bash
git clone https://github.com/cle15102005/deepfake-detection.git
cd deepfake-detection
```

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Required Packages

```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
mtcnn>=0.1.1
Pillow>=8.3.0
```

---

## 💻 Usage

### Quick Start - Inference

```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('models/efficientnetb0_best.keras')

# Load and preprocess image
img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize

# Predict
prediction = model.predict(img_array)
result = "FAKE" if prediction[0][0] > 0.5 else "REAL"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

### Data Preprocessing

```bash
# Extract faces from videos using MTCNN
python scripts/preprocess_data.py \
    --input_dir data/videos/ \
    --output_dir data/processed/ \
    --frame_interval 5
```

### Training from Scratch

```bash
# Train EfficientNetB0
python train.py \
    --model efficientnet \
    --data_dir data/Normal_Dataset/ \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001

# Train Xception
python train.py \
    --model xception \
    --data_dir data/Normal_Dataset/ \
    --epochs 50 \
    --batch_size 32

# Train MesoNet
python train.py \
    --model mesonet \
    --data_dir data/Normal_Dataset/ \
    --epochs 50 \
    --batch_size 32
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --model_path models/efficientnetb0_best.keras \
    --test_dir data/Normal_Dataset/test/ \
    --output_dir results/
```

---

## 📁 Project Structure

```
deepfake-detection/
│
├── data/
│   ├── FINAL_DATASET/
│   │   ├── Normal_Dataset/
│   │   └── Augmented_Dataset/
│   └── models/
│
├── src/
│   ├── models/
│   │   ├── efficientnet.py
│   │   ├── xception.py
│   │   └── mesonet.py
│   │
│   ├── preprocessing/
│   │   ├── face_extraction.py
│   │   └── data_augmentation.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── callbacks.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       └── metrics.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train_all_models.sh
│   └── evaluate_models.sh
│
├── results/
│   ├── plots/
│   ├── confusion_matrices/
│   └── classification_reports/
│
├── docs/
│   ├── project_report.pdf
│   └── presentation.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🎓 Training

### Training Configuration

```python
# Hyperparameters (unified across models)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
IMAGE_SIZE = (224, 224)

# EfficientNetB0 & Xception
DROPOUT_RATE = 0.2
UNFREEZE_LAYERS = 0.30  # Unfreeze last 30%

# MesoNet
DROPOUT_RATE = 0.5  # Aggressive regularization
```

### Callbacks

1. **Early Stopping**: `monitor=val_loss, patience=10`
2. **Model Checkpoint**: `monitor=val_auc, mode=max`
3. **Learning Rate Scheduler**: `ReduceLROnPlateau(factor=0.2, patience=5)`
4. **TensorBoard**: Real-time training visualization

### Training Strategy

**Stage 1: Initial Training (10 epochs)**
- Freeze early layers
- Train custom classification head
- Learning rate: 1e-3

**Stage 2: Fine-tuning (up to 50 epochs)**
- Unfreeze last 30% layers
- Continue training with reduced LR
- Early stopping if no improvement

---

## 📈 Evaluation

### Metrics

- **Accuracy**: Overall correctness
- **AUC**: Threshold-independent discriminative power
- **F1-Score**: Balance between precision and recall
- **Precision**: Reliability of fake predictions
- **Recall**: Ability to catch all fakes
- **Loss**: Probability calibration quality

### Confusion Matrix Analysis

```python
# Generate confusion matrix and detailed metrics
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_classes)
report = classification_report(y_test, y_pred_classes, 
                               target_names=['Real', 'Fake'])
```

### Visualization

```bash
# Generate all evaluation plots
python scripts/generate_plots.py \
    --model_path models/efficientnetb0_best.keras \
    --test_dir data/Normal_Dataset/test/ \
    --output_dir results/plots/
```

Generates:
- Training history (AUC, Loss, Accuracy)
- ROC curves
- Confusion matrices
- Precision-Recall curves
- Prediction distribution histograms

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@techreport{deepfake2026,
  title={Deepfake Image Detection: A Comparative Study of EfficientNetB0, Xception, and MesoNet},
  author={Le, Viet Cuong and Dinh, Ha Hai and Tran, Trung Hieu and Hoang, Quoc Huy and Nguyen, Ngoc Linh},
  institution={Hanoi University of Science and Technology},
  year={2026},
  type={Project Report},
  note={School of Information and Communication Technology, Class IT3320E}
}
```

### Related Papers

- **EfficientNet**: Tan & Le (2019) - [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- **Xception**: Chollet (2017) - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- **MesoNet**: Afchar et al. (2018) - [MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/abs/1809.00888)
- **FaceForensics++**: Rössler et al. (2019) - [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)

---

## 👥 Contributors

**Group 6 - Deep Learning Course (IT3320E)**

| Name | Student ID | Role |
|------|-----------|------|
| Le Viet Cuong | 20235586 | Project Lead, EfficientNet Implementation |
| Dinh Ha Hai | 20235589 | Xception Implementation, Data Preprocessing |
| Tran Trung Hieu | 20235591 | MesoNet Implementation, Evaluation |
| Hoang Quoc Huy | 20235594 | Dataset Curation, Documentation |
| Nguyen Ngoc Linh | 20235599 | Experiments, Results Analysis |

**Supervisor**: Mr. Than Quang Khoat  
**Institution**: Hanoi University of Science and Technology  
**School**: Information and Communication Technology  

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FaceForensics++** team for the comprehensive benchmark dataset
- **Google/Jigsaw** for the DeepFakeDetection contribution
- **TensorFlow/Keras** teams for the excellent deep learning framework
- Our supervisor **Mr. Than Quang Khoat** for guidance and support

---

## 📧 Contact

For questions, suggestions, or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/cle15102005/deepfake-detection/issues)
- **Email**: cle15102005@gmail.com
- **Project Repository**: [https://github.com/cle15102005/deepfake-detection](https://github.com/cle15102005/deepfake-detection)

---

## 🔮 Future Work

- [ ] Cross-dataset generalization (Celeb-DF, DFDC)
- [ ] Temporal information exploitation (LSTM, 3D CNN)
- [ ] Attention mechanism visualization (Grad-CAM)
- [ ] Adversarial robustness testing
- [ ] Real-time detection optimization
- [ ] Multi-modal fusion (video + audio)
- [ ] Mobile deployment (TensorFlow Lite)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by Group 6 - HUST

</div>
