# ==========================================
# EVALUATION
# ==========================================

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_history(history, plot_dir, file_dir):
    print(" Generating training plots...")

    # Plot 1: AUC
    plt.figure(figsize=(10, 6))
    if 'auc' in history:
        plt.plot(history['auc'], label='Train AUC', linewidth=2, marker='o', markersize=4)
        plt.plot(history['val_auc'], label='Val AUC', linewidth=2, marker='s', markersize=4)
    else:
        plt.plot(history['accuracy'], label='Train Acc', linewidth=2) 
    
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
    plt.title('Model AUC - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_auc_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
    plt.title('Model Accuracy - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_accuracy_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
    plt.title('Model Loss - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_loss_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n All training plots saved successfully!\n")

def test_evaluate(model, test_dataset, plot_dir, file_dir, test_dir):
    print("\n" + "="*60)
    print(" EVALUATING MODEL ON TEST SET")
    print("="*60 + "\n")

    # Evaluate
    results = model.evaluate(test_dataset, verbose=1)

    print("\n" + "="*60)
    print(" TEST SET RESULTS")
    print("="*60)
    print(f"Loss:      {results[0]:.4f}")
    print(f"Accuracy:  {results[1]:.4f}")
    print(f"AUC:       {results[2]:.4f}")
    
    # Calculate F1 manually from Precision (idx 3) and Recall (idx 4)
    f1_score = 2 * (results[3] * results[4]) / (results[3] + results[4]) if (results[3] + results[4]) > 0 else 0
    print(f" F1 Score: {f1_score:.4f}\n")

    # Collect Predictions
    print(" Collecting predictions...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())  

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Real (0)', 'Fake (1)']

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Classification Report
    print("="*60)
    print(" DETAILED CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4)
    print(report)
    
    with open(os.path.join(test_dir, f'classification_report_{file_dir}.txt'), 'w') as f:
        f.write(report)

    # Error Analysis
    print("="*60)
    print(" ERROR ANALYSIS")
    print("="*60)

    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (Correctly identified Reals):  {tn:5d}")  
    print(f"  True Positives (Correctly identified Fakes):  {tp:5d}")  
    print(f"  False Positives (Real misclassified as Fake): {fp:5d} (False Alarm)") 
    print(f"  False Negatives (Fake misclassified as Real): {fn:5d} (Missed Detection)") 

    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
    fnr = fn/(fn+tp) if (fn+tp) > 0 else 0

    print(f"\nError Rates:")
    print(f"  False Positive Rate: {fpr:.2%}")
    print(f"  False Negative Rate: {fnr:.2%}")
    
    return results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs