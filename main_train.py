from keras import optimizers, metrics
import tensorflow as tf
import numpy as np

def train_model(model, base_model, BATCH_SIZE, LEARNING_RATE, train_ds, val_ds, callbacks, model_name):
    print("\n" + "="*60)
    print(" MODEL READY FOR TRAINING")
    print("="*60)
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("="*60 + "\n")

    print(f"Model: {model_name}")
    print(f"Epochs: 50 (with early stopping)")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")

    # 1. STRATEGY: MESONET (Train from scratch)
    if 'mesonet' in model_name:
        print(" Strategy: Full Training (From Scratch)")
        model.trainable = True
        lr = LEARNING_RATE
    
    # 2. STRATEGY: TRANSFER LEARNING (EfficientNet/Xception)
    else:
        print(" Strategy: Transfer Learning (Fine-Tuning)")
        base_model.trainable = True
        # Freeze bottom layers, keep top 30 trainable
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        lr = 1e-5 # Lower LR for fine-tuning
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc'), metrics.Precision(name='precision'), metrics.Recall(name='recall')]
    )
    
    history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)
    return history.history