import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Import utils
import preprocessing, main_eval, save

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

MODEL_PATH_TO_LOAD   = "save_efficientb0_nor/model/efficientnetb0_ffnormal_20251231_020614.keras"
HISTORY_PATH_TO_LOAD = "save_efficientb0_nor/history/efficientnetb0_ffnormal_20251231_020614.npy"

# 👇 CHECK THIS PATH (Ensure it is correct)
TEST_DATA_DIR = "TEST/TEST/test_aug" 

# ------------------------------------------
# PARAMETERS
# ------------------------------------------
MODEL_NAME      = "efficientnetb0"     
DATASET_NAME    = 'ffnormal_test_aug' 
BATCH_SIZE      = 32
IMAGE_SIZE      = (224, 224)
DROPOUT_RATE    = 0.2 
LEARNING_RATE   = 0.001 

SAVE_DIR = "save"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
file_dir     = f"{MODEL_NAME}_{DATASET_NAME}_{current_time}_REAL_VS_FAKE"

for folder in ['plot', 'test']:
    os.makedirs(os.path.join(SAVE_DIR, folder), exist_ok=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n" + "="*40)
print(" 1. CUSTOM DATA LOADING (REAL vs FAKE)")
print("="*40)

path_real = os.path.join(TEST_DATA_DIR, "real")
path_fake = os.path.join(TEST_DATA_DIR, "fake")

print(f"📂 Scanning Real: {path_real}")
files_real = glob.glob(os.path.join(path_real, "*")) 
print(f"📂 Scanning Fake: {path_fake}")
files_fake = glob.glob(os.path.join(path_fake, "*"))

if len(files_real) == 0 and len(files_fake) == 0:
    print("❌ ERROR: No images found! Check TEST_DATA_DIR.")
    exit()

# Labels: 0=Real, 1=Fake
labels_real = [0] * len(files_real)
labels_fake = [1] * len(files_fake)

test_df = pd.DataFrame({
    'filepath': files_real + files_fake,
    'label': labels_real + labels_fake
})

# Shuffle
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ⚠️ CRITICAL FIX: Ensure labels are INT, not string. 
# TensorFlow metrics need numbers.
test_df['label'] = test_df['label'].astype(int)

print(f"✅ Dataset: {len(test_df)} images (Real: {len(files_real)}, Fake: {len(files_fake)})")

# Create dummy train/val DFs because preprocessing.make_data expects them
train_df = pd.DataFrame(columns=test_df.columns)
val_df = pd.DataFrame(columns=test_df.columns)
df = pd.concat([train_df, val_df, test_df]) 

print("⚙️ Preprocessing...")
# preprocessing.make_data returns (train_ds, val_ds, test_ds)
# We only care about test_ds
_, _, test_ds = preprocessing.make_data(train_df, val_df, test_df, BATCH_SIZE, MODEL_NAME)

# ==========================================
# 2. LOAD MODEL
# ==========================================
print(f"🧠 Loading model: {MODEL_PATH_TO_LOAD}")
try:
    model = tf.keras.models.load_model(MODEL_PATH_TO_LOAD)
    print("✅ Model loaded.")
    
    # ⚠️ CRITICAL FIX: Re-compile to force specific metrics.
    # main_eval.py and save.py expect exactly 5 metrics in this order:
    # [loss, accuracy, auc, precision, recall]
    # If we don't do this, results[3] will crash.
    print("🔧 Re-compiling model to ensure metrics exist...")
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ],
        jit_compile=False
    )

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==========================================
# 3. HISTORY
# ==========================================
history = {}
if HISTORY_PATH_TO_LOAD and os.path.exists(HISTORY_PATH_TO_LOAD):
    try:
        history = np.load(HISTORY_PATH_TO_LOAD, allow_pickle=True).item()
    except: pass
for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'auc', 'val_auc']:
    if key not in history: history[key] = [] 

# ==========================================
# 4. EVALUATE
# ==========================================
print("\n" + "="*40)
print(" 3. RUNNING EVALUATION")
print("="*40)

results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs = main_eval.test_evaluate(
    model, test_ds, f"{SAVE_DIR}/plot", file_dir, f"{SAVE_DIR}/test"
)

# ==========================================
# 5. SAVE
# ==========================================
print("\n" + "="*40)
print(" 4. SAVING RESULTS")
print("="*40)

print("💾 Saving metrics...")
save.save_metrics(
    MODEL_NAME, DATASET_NAME, current_time, 
    results, f1_score, tn, fp, fn, tp, fpr, fnr, 
    history, BATCH_SIZE, LEARNING_RATE, IMAGE_SIZE, DROPOUT_RATE, 
    df, train_df, val_df, test_df, 
    f"{SAVE_DIR}/test", file_dir
)

print("💾 Saving predictions...")
save.save_pred(
    f"{SAVE_DIR}/test", file_dir, f"{SAVE_DIR}/plot", 
    y_true, y_pred, y_pred_probs, results
)

print("\n🎉 COMPLETED!")