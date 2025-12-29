import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Import utils
from utils import data_loader, preprocessing, main_eval, save
import main_train

# Import models
from models import efficientnetb0, xception, mesonet 

# ==========================================
# CONFIGURATION
# ==========================================
# OPTIONS: 'efficientnetb0', 'xception', 'mesonet'
MODEL_NAME   = "efficientnetb0" 
DATA_DIR     = "/mnt/d/PROJECT/virtual_env/DL/Project/FINAL_DATASET/Normal_Dataset/" 
SAVE_DIR     = "save"
DATASET_NAME = 'ffnormal'

# Setup Paths
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_dir     = f"{MODEL_NAME}_{DATASET_NAME}_{current_time}"
for folder in ['model', 'history', 'plot', 'test']:
    os.makedirs(os.path.join(SAVE_DIR, folder), exist_ok=True)

# Hyperparams
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
DROPOUT_RATE = 0.5 if MODEL_NAME == 'xception' else 0.2

# 1. LOAD DATA
train_df, val_df, test_df = data_loader.load_datasets(DATA_DIR)
data_loader.check_imbalance(train_df, "Train")

# Reconstruct full df for save.py statistics
df = pd.concat([train_df, val_df, test_df])

# 2. PREPROCESSING
train_ds, val_ds, test_ds = preprocessing.make_data(train_df, val_df, test_df, BATCH_SIZE, MODEL_NAME)

# 3. BUILD MODEL
if MODEL_NAME == 'efficientnetb0':
    module = efficientnetb0
elif MODEL_NAME == 'xception':
    module = xception
elif MODEL_NAME == 'mesonet':
    module = mesonet

model, base_model = module.create_model((224, 224, 3), DROPOUT_RATE, 1)

# 4. CALLBACKS
# We can use efficientnetb0's callback setup as a generic helper
callbacks_list = efficientnetb0.set_callbacks(f"{SAVE_DIR}/model", file_dir, SAVE_DIR, current_time)

# 5. TRAIN
history = main_train.train_model(model, base_model, BATCH_SIZE, LEARNING_RATE, 
                                 train_ds, val_ds, callbacks_list, MODEL_NAME)

# 6. EVALUATE
np.save(os.path.join(SAVE_DIR, 'history', f"{file_dir}.npy"), history)
main_eval.plot_history(history, f"{SAVE_DIR}/plot", file_dir)

# Run Test Evaluation
results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs = main_eval.test_evaluate(
    model, test_ds, f"{SAVE_DIR}/plot", file_dir, f"{SAVE_DIR}/test"
)

# SAVE METRICS & PREDICTIONS
print(" Saving metrics and predictions...")
save.save_metrics(
    MODEL_NAME, DATASET_NAME, current_time, 
    results, f1_score, tn, fp, fn, tp, fpr, fnr, 
    history, BATCH_SIZE, LEARNING_RATE, IMAGE_SIZE, DROPOUT_RATE, 
    df, train_df, val_df, test_df, 
    f"{SAVE_DIR}/test", file_dir
)

save.save_pred(
    f"{SAVE_DIR}/test", file_dir, f"{SAVE_DIR}/plot", 
    y_true, y_pred, y_pred_probs, results
)

print(" DONE.")