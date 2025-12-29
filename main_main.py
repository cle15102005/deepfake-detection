import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from datetime import datetime

#import files
from utils import data_loader, preprocessing, main_eval, save
from models import efficientnetb0, xception, mesonet
import main_train

# Data directory
DATA_DIR    = "/mnt/d/PROJECT/virtual_env/DL/Project/FINAL_DATASET/Normal_Dataset/" 
# Saving directory
SAVE_DIR    = "save"
MODEL_NAME  = "efficientnetb0"
DATASET_NAME= 'ffnormal'

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir   = f"{SAVE_DIR}/model"
history_dir = f"{SAVE_DIR}/history"
plot_dir    = f"{SAVE_DIR}/plot"
test_dir    = f"{SAVE_DIR}/test"
file_dir    = f"{MODEL_NAME}_{DATASET_NAME}_{current_time}"

# Make dirs
os.makedirs(SAVE_DIR, exist_ok=True) 
os.makedirs(model_dir, exist_ok=True) 
os.makedirs(history_dir, exist_ok=True) 
os.makedirs(plot_dir, exist_ok=True) 
os.makedirs(test_dir, exist_ok=True) 

# ========================================== #
INPUT_SHAPE = (224, 224, 3)
IMAGE_SIZE = (224, 224)  
NUM_CLASSES = 1  # Binary classification (fake vs real)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DROPOUT_RATE = 0.2

# LOAD DATA FRAME
filepaths, labels = data_loader.scan(DATA_DIR)
df = data_loader.make_df(filepaths, labels)
data_loader.check_imbalance(df)
train_df, val_df, test_df = data_loader.split_data(df)

# MAKE DATA (PREPROCESSING INCLUDED)
train_dataset, validation_dataset, test_dataset = preprocessing.make_data(train_df, val_df, test_df, BATCH_SIZE)
preprocessing.verify_pipeline(train_dataset, plot_dir, file_dir)

# CONFIGURE MODEL
model, base_model = efficientnetb0.create_model(INPUT_SHAPE, DROPOUT_RATE, NUM_CLASSES)
early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback = efficientnetb0.set_callbacks(model_dir, file_dir, 
                                                                                                  SAVE_DIR, current_time)
callbacks=[
            early_stopping,
            model_checkpoint,
            lr_scheduler,  
            tensorboard_callback
        ]

# TRAIN MODEL
history = main_train.train_model(model, base_model, BATCH_SIZE, LEARNING_RATE,
                                 train_dataset, validation_dataset, callbacks)
    
# Save history
history_path = os.path.join(history_dir, f"{file_dir}.npy")
np.save(history_path, history) 
print(f" Training history saved: {history_path}")
print(f"Total Epochs:   {len(history['loss'])}") 
 
# EVALUATE ON TEST DATASET
main_eval.plot_history(history, plot_dir, file_dir)
results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs = main_eval.test_evaluate(
    model, test_dataset, plot_dir, file_dir, test_dir)

# SAVE ALL
save.save_metrics(MODEL_NAME, DATASET_NAME, current_time, results, f1_score, tn, fp, fn, tp, fpr, fnr, 
                 history, BATCH_SIZE, LEARNING_RATE, IMAGE_SIZE, DROPOUT_RATE, df, train_df, val_df, test_df, test_dir, file_dir)

save.save_pred(test_dir, file_dir, plot_dir, y_true, y_pred, y_pred_probs, results)

# ==========================================
# SUMMARY REPORT
# ==========================================

print("\n" + "="*60)
print("  EVALUATION SUMMARY")
print("="*60)
print(f"\nModel: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Timestamp: {current_time}")
print(f"\n  Test Set Performance:")
print(f"  Accuracy:  {results[1]*100:.2f}%")
print(f"  AUC:       {results[2]:.4f}")
print(f"  Precision: {results[3]:.4f}")
print(f"  Recall:    {results[4]:.4f}")
print(f"  F1 Score:  {f1_score:.4f}")
print("="*60 + "\n")