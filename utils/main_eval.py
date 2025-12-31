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
    print(" Generating training plots...")

    # Plot 1: AUC 
    plt.figure(figsize=(10, 6))
    if 'auc' in history:
        plt.plot(history['auc'], label='Train AUC', linewidth=2, marker='o', markersize=4)
        plt.plot(history['val_auc'], label='Val AUC', linewidth=2, marker='s', markersize=4)
    else:
        # Fallback if metric name varies
        keys = [k for k in history.keys() if 'auc' in k]
        if keys:
            plt.plot(history[keys[0]], label='Train AUC', linewidth=2)
            plt.plot(history['val_'+keys[0]], label='Val AUC', linewidth=2)
        else:
             plt.plot(history['accuracy'], label='Train Acc', linewidth=2) # Fallback to accuracy

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

    print(f" All training plots saved successfully!\n")

def test_evaluate(model, test_dataset, plot_dir, file_dir, test_dir):
    """
    Phase 2: Benchmarking
    """
    print("\n" + "="*60)
    print(" EVALUATING MODEL ON TEST SET")
    print("="*60 + "\n")

    # Evaluate
    # Results will be [loss, accuracy, auc, precision, recall]
    results = model.evaluate(test_dataset, verbose=1)

    # Safe unpacking
    precision = results[3] if len(results) > 3 else 0.0
    recall = results[4] if len(results) > 4 else 0.0
    
    # Calculate F1 Score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Collect Predictions
    print(" Collecting predictions...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_dataset:
        y_true.extend(labels.numpy().flatten())
        
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())  

    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle cases where CM is not 2x2 (e.g. only one class in test set)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        print("Warning: Confusion Matrix is not 2x2. Dataset might be unbalanced.")
        # Fallback logic
        tn, fp, fn, tp = 0, 0, 0, 0
        try:
            tn = cm[0,0] # if only negatives exist
        except: pass

    # Calculate Error Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Plot Confusion Matrix
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

    # 2. Classification Report
    print("="*60)
    print(" DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    # Use try-except in case y_true has only 1 class
    try:
        report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4)
        print(report)
        with open(os.path.join(test_dir, f'classification_report_{file_dir}.txt'), 'w') as f:
            f.write(report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (Correctly identified Reals):  {tn:5d}")  
    print(f"  True Positives (Correctly identified Fakes):  {tp:5d}")  
    print(f"  False Positives (Real misclassified as Fake): {fp:5d} (False Alarm)") 
    print(f"  False Negatives (Fake misclassified as Real): {fn:5d} (Missed Detection)") 
    
    return results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs