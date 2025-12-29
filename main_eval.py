# ==========================================
# EVALUATION
# ==========================================

# Import modules
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Plot history
def plot_history(history, plot_dir, file_dir):
    print(" Generating training plots...\n")

    # Plot 1: AUC
    plt.figure(figsize=(10, 6))
    plt.plot(history['auc'], label='Train AUC', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_auc'], label='Val AUC', linewidth=2, marker='s', markersize=4)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
    plt.title('Model AUC - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_auc_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Plot 1 saved: auc")

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
    print(f" Plot 2 saved: accuracy")

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
    print(f" Plot 3 saved: loss")

    # Plot 4: Precision
    plt.figure(figsize=(10, 6))
    plt.plot(history['precision'], label='Train Precision', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_precision'], label='Val Precision', linewidth=2, marker='s', markersize=4)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
    plt.title('Model Precision - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_precision_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Plot 4 saved: precision")

    # Plot 5: Recall
    plt.figure(figsize=(10, 6))
    plt.plot(history['recall'], label='Train Recall', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_recall'], label='Val Recall', linewidth=2, marker='s', markersize=4)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
    plt.title('Model Recall - Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_recall_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Plot 5 saved: recall")

    # Plot 6: Learning Rate
    plt.figure(figsize=(10, 6))
    if 'lr' in history:
        plt.plot(history['lr'], linewidth=2, color='purple', marker='D', markersize=5)
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Fine-tune starts')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Logged\n\n(Enable by adding lr to metrics)',
                ha='center', va='center', fontsize=14, color='gray')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'plot_lr_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Plot 6 saved: learning_rate")

    print(f"\n All 6 training plots saved successfully!\n")

# Evaluate on testset
def test_evaluate(model, test_dataset, plot_dir, file_dir, test_dir):
    print("\n" + "="*60)
    print(" EVALUATING MODEL ON TEST SET")
    print("="*60 + "\n")

    # Evaluate on test set
    print("Running evaluation on test dataset...\n")
    results = model.evaluate(test_dataset, verbose=1)

    print("\n" + "="*60)
    print(" TEST SET RESULTS")
    print("="*60)
    print(f"Loss:      {results[0]:.4f}")
    print(f"Accuracy:  {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"AUC:       {results[2]:.4f}")
    print(f"Precision: {results[3]:.4f}")
    print(f"Recall:    {results[4]:.4f}")
    print("="*60 + "\n")
    
    # Calculate F1 Score
    f1_score = 2 * (results[3] * results[4]) / (results[3] + results[4]) if (results[3] + results[4]) > 0 else 0
    print(f" F1 Score: {f1_score:.4f}\n")

    # ==========================================
    # COLLECT PREDICTIONS
    # ==========================================

    print(" Collecting predictions from test set...")

    y_true = []
    y_pred_probs = []

    for images, labels in test_dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())  

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(f" Collected {len(y_true)} predictions\n")

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    print(" Creating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Fake (0)', 'Real (1)']

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)

    # Add accuracy annotation
    total = cm.sum()
    accuracy = (cm[0,0] + cm[1,1]) / total
    plt.text(1, -0.35, f'Overall Accuracy: {accuracy:.2%}',
            ha='center', fontsize=13, fontweight='bold',
            transform=plt.gca().transAxes)

    plt.tight_layout()
    confusion_matrix_path = os.path.join(plot_dir, f'confusion_matrix_{file_dir}.png')
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Confusion matrix saved: {confusion_matrix_path}\n")

    # ==========================================
    # DETAILED CLASSIFICATION REPORT
    # ==========================================

    print("="*60)
    print(" DETAILED CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'], digits=4)
    print(report)

    # Save classification report to file
    report_path = os.path.join(test_dir, f'classification_report_{file_dir}.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT - TEST SET\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write("="*60 + "\n")

    print(f"\n Classification report saved: {report_path}\n")

    # ==========================================
    # ERROR ANALYSIS
    # ==========================================

    print("="*60)
    print(" ERROR ANALYSIS")
    print("="*60)

    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (Correctly identified fakes):  {tn:5d}")
    print(f"  True Positives (Correctly identified reals):  {tp:5d}")
    print(f"  False Positives (Real misclassified as real): {fp:5d}")
    print(f"  False Negatives (Fake misclassified as fake): {fn:5d}")

    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
    fnr = fn/(fn+tp) if (fn+tp) > 0 else 0

    print(f"\nError Rates:")
    print(f"  False Positive Rate: {fpr:.2%}")
    print(f"  False Negative Rate: {fnr:.2%}")
    
    return results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs