# ==========================================
# TRAINING
# ==========================================

# Import modules
from keras import optimizers, metrics
import tensorflow as tf

def train_model(model, base_model, BATCH_SIZE, LEARNING_RATE, train_dataset, validation_dataset, callbacks):
    print("\n" + "="*60)
    print(" MODEL READY FOR TRAINING")
    print("="*60)
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"Non-trainable Parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
    print("="*60 + "\n")

    print("\n Ready to train!")

    print("\n" + "="*60)
    print(" STARTING MODEL TRAINING")
    print("="*60)
    print(f"Epochs: 50 (with early stopping)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print("="*60 + "\n")

    # Unfreeze the layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Compile 
    fine_tune_lr = 1e-5
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss='binary_crossentropy',  
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    # 3. Fitting 
    print("---Fine-tuning layers---")
    history = model.fit(
        train_dataset,
        epochs=50,  # Total number of epochs
        initial_epoch=0, 
        validation_data=validation_dataset,
        callbacks= callbacks
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60 + "\n")

    return history.history