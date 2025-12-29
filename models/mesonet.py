# ==========================================
# MESONET CONFIGURATION
# ==========================================
from keras import layers, models, regularizers

def create_model(INPUT_SHAPE, DROPOUT_RATE, NUM_CLASSES):
    print("  Building MesoNet (Meso4) model...")

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = inputs

    # Block 1
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Block 2
    x = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Block 3
    x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Block 4
    x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Classification Head
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(16)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MesoNet")
    
    # MesoNet is trained from scratch, so 'base_model' is just the model itself
    return model, model