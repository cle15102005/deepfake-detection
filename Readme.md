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
import tensorflow as tf
from utils.preprocessing import preprocess_image

# Load trained model
model = tf.keras.models.load_model('save/model/efficientnetb0_ffnormal_TIMESTAMP.keras')

# Preprocess and predict single image
img_path = 'test_image.jpg'
img_array = preprocess_image(img_path, model_name='efficientnetb0')

# Get prediction
prediction = model.predict(img_array)
result = "FAKE" if prediction[0][0] > 0.5 else "REAL"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

### Using the Complete Pipeline

```python
# Run the entire training and evaluation pipeline
python main_main.py
```

### Data Preprocessing

Your data should be organized as:
```
FINAL_DATASET/
└── Normal_Dataset/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

The preprocessing pipeline in `utils/preprocessing.py` handles:
- Dynamic preprocessing based on model (EfficientNet, Xception, MesoNet)
- Automatic resizing to 224×224
- Model-specific normalization

### Training from Scratch

**Full Pipeline (Recommended)**
```python
# Edit configuration in main_main.py
MODEL_NAME = "efficientnetb0"  # or "xception", "mesonet"
DATA_DIR = "path/to/FINAL_DATASET/Normal_Dataset/"

# Run complete training + evaluation
python main_main.py
```

This will:
1. Load and preprocess data
2. Build the selected model
3. Train with automatic callbacks
4. Evaluate on test set
5. Generate all plots and metrics
6. Save everything to `save/` directory

**Training Only (Advanced)**
```python
from main_train import train_model
from models import efficientnetb0

# Create model
model, base_model = efficientnetb0.create_model((224, 224, 3), 0.2, 1)

# Train
history = train_model(model, base_model, 32, 0.001, 
                     train_ds, val_ds, callbacks_list, 'efficientnetb0')
```

---

## 📁 Project Structure

```
deepfake-detection/
│
├── models/                      # Model architecture implementations
│   ├── efficientnetb0.py       # EfficientNetB0 architecture
│   ├── xception.py             # Xception architecture
│   └── mesonet.py              # MesoNet architecture
│
├── utils/                       # Utility scripts
│   ├── data_loader.py          # Dataset loading and batching
│   ├── main_eval.py            # Model evaluation script
│   ├── preprocessing.py        # Image preprocessing and face extraction
│   └── save.py                 # Model checkpoint saving utilities
│
├── main_train.py               # Main training script
├── main_main.py                # Main execution pipeline
├── Readme.md                   # This file
└── requirements.txt            # Python dependencies
```

### Quick Navigation

- **Training**: Use `main_train.py` to train models
- **Evaluation**: Use `utils/main_eval.py` for testing
- **Models**: All three architectures in `models/` directory
- **Data Processing**: `utils/preprocessing.py` handles face extraction

---

## 🎓 Training

### Configuration (in `main_main.py`)

```python
# Model Selection
MODEL_NAME = "efficientnetb0"  # Options: 'efficientnetb0', 'xception', 'mesonet'

# Data Path
DATA_DIR = "/path/to/FINAL_DATASET/Normal_Dataset/"

# Hyperparameters (unified across models)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
DROPOUT_RATE = 0.5 if MODEL_NAME == 'mesonet' else 0.2
```

### Training Strategy

The `main_train.py` automatically handles two different strategies:

**For MesoNet (From Scratch):**
```python
# All layers trainable from start
model.trainable = True
learning_rate = 0.001
```

**For EfficientNetB0 & Xception (Transfer Learning):**
```python
# Freeze bottom layers, unfreeze top 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
learning_rate = 1e-5  # Lower LR for fine-tuning
```

### Callbacks (Automatic)

The system uses callbacks from `efficientnetb0.py` (generic for all models):

1. **Early Stopping**: `monitor=val_loss, patience=10`
2. **Model Checkpoint**: `monitor=val_auc, mode=max` (saves best model)
3. **Learning Rate Scheduler**: `ReduceLROnPlateau(factor=0.2, patience=5)`
4. **TensorBoard**: Real-time training visualization

### Output Structure

After training, `main_main.py` generates:
```
save/
├── model/
│   └── {model}_{dataset}_{timestamp}.keras
├── history/
│   └── {model}_{dataset}_{timestamp}.npy
├── plot/
│   ├── plot_auc_{timestamp}.png
│   ├── plot_accuracy_{timestamp}.png
│   ├── plot_loss_{timestamp}.png
│   ├── confusion_matrix_{timestamp}.png
│   ├── roc_curve_{timestamp}.png
│   ├── precision_recall_curve_{timestamp}.png
│   └── prediction_distribution_{timestamp}.png
├── test/
│   ├── test_metrics_{timestamp}.json
│   ├── test_predictions_{timestamp}.npz
│   └── classification_report_{timestamp}.txt
└── logs/
    └── fit_{timestamp}/  # TensorBoard logs
```

---

## 📈 Evaluation

### Automatic Evaluation (via main_main.py)

The pipeline automatically:
1. Loads test dataset from `DATA_DIR/test/`
2. Runs model evaluation (loss, accuracy, AUC, precision, recall)
3. Calculates F1-score
4. Generates confusion matrix
5. Creates ROC and PR curves
6. Saves all metrics to JSON

All evaluation happens automatically when you run:
```bash
python main_main.py
```

### Custom Evaluation

To evaluate a saved model on new test data:

```python
import tensorflow as tf
from utils import data_loader, preprocessing, main_eval

# Load model
model = tf.keras.models.load_model('save/model/your_model.keras')

# Load test data
test_df = data_loader.scan_folder('path/to/test_data/')
test_ds = preprocessing.create_ds(test_df, batch_size=32)

# Evaluate
results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs = \
    main_eval.test_evaluate(model, test_ds, 'save/plot', 'custom_test', 'save/test')
```

### Metrics Explanation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **AUC** | Area under ROC curve | Threshold-independent performance |
| **Precision** | TP/(TP+FP) | Reliability of fake predictions |
| **Recall** | TP/(TP+FN) | Ability to catch all fakes |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Balanced performance |

**Note:** In our implementation:
- Label `0` = Real images
- Label `1` = Fake images
- TP = Correctly identified fakes
- TN = Correctly identified reals

### Visualization

The `utils/main_eval.py` automatically generates:

1. **Training History Plots**
   - AUC progression
   - Accuracy progression  
   - Loss progression

2. **Test Evaluation Plots** (via `save.py`)
   - Confusion matrix with error breakdown
   - ROC curve (TPR vs FPR)
   - Precision-Recall curve
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
