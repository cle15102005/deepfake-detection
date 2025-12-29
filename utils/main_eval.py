import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_history(history, plot_dir, file_dir):
    """
    Phase 1: Training - Tuning
    Generates plots for Loss, Accuracy, and AUC to verify convergence and overfitting.
    """
    # Plot 1: AUC 
    plt.figure(figsize=(10, 6))
    if 'auc' in history:
        plt.plot(history['auc'], label='Train AUC', linewidth=2, marker='o', markersize=4)
        plt.plot(history['val_auc'], label='Val AUC', linewidth=2, marker='s', markersize=4)
    else:
        # Fallback for metric naming variations
        keys = [k for k in history.keys() if 'auc' in k]
        if keys:
            plt.plot(history[keys[0]], label='Train AUC', linewidth=2)
            plt.plot(history['val_'+keys[0]], label='Val AUC', linewidth=2)

    plt.title('Model AUC - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_auc_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    plt.title('Model Accuracy - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_accuracy_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
    plt.title('Model Loss - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_loss_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def test_evaluate(model, test_dataset, plot_dir, file_dir, test_dir):
    """
    Phase 2: Benchmarking
    Calculates Accuracy, AUC, F1, and generates the Confusion Matrix.
    """
    # Evaluate (Silent mode)
    results = model.evaluate(test_dataset, verbose=0)
    
    # Calculate F1 Score manually (Precision=idx 3, Recall=idx 4)
    precision = results[3]
    recall = results[4]
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Predictions
    y_true = []
    y_pred_probs = []
    for images, labels in test_dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())  

    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real (0)', 'Fake (1)'], 
                yticklabels=['Real (0)', 'Fake (1)'],
                cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Save Classification Report (Contains F1, Accuracy, Recall, Precision)
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4)
    with open(os.path.join(test_dir, f'classification_report_{file_dir}.txt'), 'w') as f:
        f.write(report)
    
    # Return metrics for main loop usage if needed: [Loss, Accuracy, AUC, Precision, Recall], F1
    return results, f1_score