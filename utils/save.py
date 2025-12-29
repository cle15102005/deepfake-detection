# ==========================================
# SAVE ALL METRICS TO JSON
# ==========================================

# Import modules
import json, os
import numpy as np
import matplotlib.pyplot as plt

def save_metrics(MODEL_NAME, DATASET_NAME, current_time, results, f1_score, tn, fp, fn, tp, fpr, fnr, 
                 history, BATCH_SIZE, LEARNING_RATE, IMAGE_SIZE, DROPOUT_RATE, df, train_df, val_df, test_df, test_dir, file_dir):
    metrics_dict = {
        'model_name': MODEL_NAME,
        'dataset_name': DATASET_NAME,
        'timestamp': current_time,
        'test_metrics': {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'auc': float(results[2]),
            'precision': float(results[3]),
            'recall': float(results[4]),
            'f1_score': float(f1_score)
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'error_rates': {
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr)
        },
        'training_info': {
            'total_epochs': len(history['loss']),
            'batch_size': BATCH_SIZE,
            'initial_learning_rate': LEARNING_RATE,
            'fine_tune_learning_rate': 1e-5,
            'image_size': IMAGE_SIZE,
            'dropout_rate': DROPOUT_RATE
        },
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'fake_count': int((df['label'] == 1).sum()), # 1 = Fake
            'real_count': int((df['label'] == 0).sum())  # 0 = Real
        }
    }

    metrics_path = os.path.join(test_dir, f'test_metrics_{file_dir}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"\n Test metrics saved: {metrics_path}")
    
# ==========================================
# SAVE PREDICTIONS
# ==========================================
def save_pred(test_dir, file_dir, plot_dir, y_true, y_pred, y_pred_probs, results):
    predictions_path = os.path.join(test_dir, f'test_predictions_{file_dir}.npz')
    np.savez(predictions_path, 
            y_true=y_true, 
            y_pred=y_pred, 
            y_pred_probs=y_pred_probs)
    print(f" Test predictions saved: {predictions_path}")

    # ==========================================
    # ROC CURVE
    # ==========================================

    from sklearn.metrics import roc_curve, auc

    print("\n Generating ROC curve...")

    fpr_roc, tpr_roc, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr_roc, tpr_roc)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
            fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(plot_dir, f'roc_curve_{file_dir}.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" ROC curve saved: {roc_path}")

    # ==========================================
    # PRECISION-RECALL CURVE
    # ==========================================

    from sklearn.metrics import precision_recall_curve, average_precision_score

    print(" Generating Precision-Recall curve...")

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)

    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='green', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    plt.axhline(y=results[3], color='red', linestyle='--', lw=1.5,
                label=f'Model Precision @ 0.5 threshold = {results[3]:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pr_path = os.path.join(plot_dir, f'precision_recall_curve_{file_dir}.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Precision-Recall curve saved: {pr_path}")

    # ==========================================
    # PREDICTION DISTRIBUTION
    # ==========================================

    print(" Generating prediction distribution plot...")

    plt.figure(figsize=(12, 6))

    # Plot for fake samples (label 1)
    plt.subplot(1, 2, 1)
    fake_probs = y_pred_probs[y_true == 1]  # <--- Correct: 1 is Fake
    plt.hist(fake_probs, bins=50, color='red', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    plt.xlabel('Predicted Probability', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.title('Distribution of Predictions for FAKE Images', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot for real samples (label 0)
    plt.subplot(1, 2, 2)
    real_probs = y_pred_probs[y_true == 0]  # <--- FIXED: 0 is Real (Was 1)
    plt.hist(real_probs, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    plt.xlabel('Predicted Probability', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.title('Distribution of Predictions for REAL Images', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(plot_dir, f'prediction_distribution_{file_dir}.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Prediction distribution saved: {dist_path}")