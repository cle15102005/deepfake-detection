# ==========================================
# XCEPTION CONFIGURATION
# ==========================================

# Import modules
from keras import applications, models, layers, callbacks
import os

# Create model
def create_model(INPUT_SHAPE, DROPOUT_RATE, NUM_CLASSES):
    print("  Building Xception model for deepfake detection...\n")

    # Load Pre-trained Base Model
    print(" Loading Xception with ImageNet weights...")

    base_model = applications.Xception(
        include_top= False,
        weights="imagenet",
        input_tensor=None,
        input_shape= INPUT_SHAPE,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        name="xception",
    )

    # Freeze all layers initially
    base_model.trainable = False

    print(f"  Base model loaded: {len(base_model.layers)} layers")
    print(f"  All layers frozen for transfer learning\n")

    # Build Custom Classification Head
    print("  Adding custom classification layers...")

    # Base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_2')(x)

    output = layers.Dense(NUM_CLASSES, activation='sigmoid', name='output')(x)

    # Create final model
    model = models.Model(inputs=base_model.input, outputs=output, name='Xception_DeepfakeDetector')

    print(f" Custom head added: GlobalAvgPool → Dropout({DROPOUT_RATE}) → Dense(1, sigmoid)\n")

    return model, base_model    

# Setup Callbacks
def set_callbacks(model_dir, file_dir, SAVE_DIR, current_time):
    save_path = os.path.join(model_dir, f"{file_dir}.keras")

    print(" Setting up training callbacks...\n")
    # 1. Early Stopping - prevents overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',              # Monitor validation AUC
        patience=10,                    # Wait 10 epochs before stopping
        mode='max',                     # We want to maximize AUC
        restore_best_weights=True,      # Restore weights from best epoch
        verbose=1
    )
    print(" Early Stopping: patience=10, monitor=val_auc")

    # 2. Model Checkpoint - saves best model
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor='val_loss',
        mode='max',
        save_best_only=True,            # Only save when val_loss improves
        verbose=1
    )
    print(f" Model Checkpoint: {save_path}")

    # 3. Learning Rate Scheduler - reduces LR when plateau
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.2,                     # New LR = old LR * 0.2
        patience=5,                     # Wait 5 epochs before reducing
        min_lr=1e-7,                    # Don't go below this LR
        verbose=1
    )
    print(" LR Scheduler: factor=0.2, patience=5, min_lr=1e-7")

    # 4. TensorBoard - for visualization
    log_dir = os.path.join(SAVE_DIR, "logs", f"fit_{current_time}")
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,               # Log weight histograms every epoch
        write_graph=True,
        update_freq='epoch'
    )
    print(f" TensorBoard: {log_dir}")
    
    return early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback