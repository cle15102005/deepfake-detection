import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import applications
from sklearn.metrics import f1_score

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_PATH    = "save_efficientb0_nor/model/efficientnetb0_ffnormal_20251231_020614.keras"
TEST_DATA_DIR = "TEST/TEST/test_aug" # Folder containing 'real' and 'fake' subfolders
BATCH_SIZE    = 32
IMAGE_SIZE    = (224, 224)

# ==========================================
# 1. SIMPLE DATA LOADER
# ==========================================
def build_test_dataset(test_dir, batch_size, image_size):
    # Find files
    path_real = os.path.join(test_dir, "real")
    path_fake = os.path.join(test_dir, "fake")
    
    files_real = glob.glob(os.path.join(path_real, "*"))
    files_fake = glob.glob(os.path.join(path_fake, "*"))
    
    if not files_real and not files_fake:
        raise ValueError(f"No images found in {test_dir}. Check paths.")

    # Create DataFrame: 0=Real, 1=Fake
    df = pd.DataFrame({
        'filepath': files_real + files_fake,
        'label': [0]*len(files_real) + [1]*len(files_fake)
    })
    
    # Define Image Loading
    def process_path(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        # EfficientNet Preprocessing (built-in or specific)
        img = applications.efficientnet.preprocess_input(img) 
        return img, label

    # Build Dataset
    ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['label'].values))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Loaded Test Data: {len(df)} images")
    return ds, df['label'].values

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # A. Load Data
    test_ds, y_true_all = build_test_dataset(TEST_DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

    # B. Load Model
    print(f"🧠 Loading Model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # C. Compile with specific metrics (Loss, Acc, AUC)
    print("🔧 Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # D. Run Evaluation (Gets Loss, Acc, AUC)
    print("🚀 Evaluating...")
    results = model.evaluate(test_ds, verbose=1)
    loss, acc, auc_score = results[0], results[1], results[2]

    # E. Calculate F1 Score (Needs Predictions)
    print("🔮 Calculating F1 Score...")
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Ensure y_true matches y_pred shape (handle last batch drop if any, usually fine here)
    # y_true_all comes from dataframe, y_pred comes from dataset iteration order.
    # Note: tf.data without shuffle preserves order, so this is safe.
    f1 = f1_score(y_true_all, y_pred_classes)

    # ==========================================
    # 3. OUTPUT RESULTS
    # ==========================================
    print("\n" + "="*30)
    print(" 🏆 FINAL TEST RESULTS")
    print("="*30)
    print(f" Loss:      {loss:.4f}")
    print(f" Accuracy:  {acc:.4f}")
    print(f" AUC:       {auc_score:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    print("="*30)

    # Save to simple JSON
    output_data = {
        "loss": loss,
        "accuracy": acc,
        "auc": auc_score,
        "f1_score": f1
    }
    
    with open("test_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("✅ Results saved to 'test_results.json'")