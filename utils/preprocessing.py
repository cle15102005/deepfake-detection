# ==========================================
# PREPROCESSING
# ==========================================

import os
import tensorflow as tf
from keras import applications
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

def load_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img, label

def preprocess_image(img, label, model_name):
    # Resize
    img = tf.image.resize(img, (224, 224))
    
    # Dynamic Preprocessing based on Model
    if 'efficientnet' in model_name:
        img = applications.efficientnet.preprocess_input(img)
    elif 'xception' in model_name:
        img = applications.xception.preprocess_input(img)
    elif 'mesonet' in model_name:
        img = img / 255.0 # Rescale [0, 1]
        
    return img, label

def make_data(train_df, val_df, test_df, BATCH_SIZE, model_name='efficientnetb0'):
    print(f" Creating data pipeline for: {model_name}")

    # Wrapper to pass model_name to map
    def preprocess_wrapper(img, label):
        return preprocess_image(img, label, model_name)

    def create_ds(df, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['label'].values))
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
        ds = ds.map(preprocess_wrapper, num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_ds = create_ds(train_df, shuffle=True)
    val_ds   = create_ds(val_df)
    test_ds  = create_ds(test_df)

    return train_ds, val_ds, test_ds

# Verify the pipeline
def verify_pipeline(train_dataset, plot_dir, file_dir):
    print(" Verifying data pipeline...\n")

    # Get one batch
    image_batch, label_batch = next(iter(train_dataset))

    print(f"  Batch shape: {image_batch.shape}")
    print(f"  Expected: (32, 224, 224, 3)")

    # Check pixel value range
    min_val = tf.reduce_min(image_batch).numpy()
    max_val = tf.reduce_max(image_batch).numpy()
    mean_val = tf.reduce_mean(image_batch).numpy()

    print(f"\n Pixel value statistics:")
    print(f"  Min:  {min_val:.2f}")
    print(f"  Max:  {max_val:.2f}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  (EfficientNet preprocessing normalizes values, so these won't be 0-255)")

    # Check labels
    print(f"\n Label batch shape: {label_batch.shape}")
    print(f"  Sample labels: {label_batch[:5].numpy()}")

    # Visualize sample images
    print("\n Visualizing sample images...\n")

    # Grab a batch for visualization
    vis_images, vis_labels = next(iter(train_dataset))

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()

    for idx in range(8):
        img = vis_images[idx].numpy()
        label = vis_labels[idx].numpy()

        # Denormalize for visualization
        img_denorm = (img - img.min()) / (img.max() - img.min())

        axes[idx].imshow(img_denorm)
        axes[idx].set_title(f"Label: {'Real' if label == 0 else 'Fake'}",
                        fontsize=12,
                        color='green' if label == 0 else 'red')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Sample Images from Training Set (After Preprocessing)',
                fontsize=14, y=1.02)
    plt.savefig(os.path.join(plot_dir, f'verify_pipeline_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()