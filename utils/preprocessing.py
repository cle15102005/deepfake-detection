# ==========================================
# PREPROCESSING
# ==========================================

#import modules
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import applications
import os   

# Configuration
AUTOTUNE = tf.data.AUTOTUNE

print(" Setting up EfficientNet preprocessing pipeline...")
def load_image(filepath, label):
    """
    Load and decode image file.
    Returns raw image tensor and label.
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return img, label

def is_valid_image(img, label):
    """
    Filter out corrupted images with invalid dimensions.
    Returns True only if both height and width > 0.
    """
    shape = tf.shape(img)
    return (shape[0] > 0) & (shape[1] > 0)

def preprocess_image(img, label):
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (224, 224))
    
    #if 'efficientnet':
    img = applications.efficientnet.preprocess_input(img)
    #elif 'xception' 
    #img = applications.xception.preprocess_input(img)
    #elif 'mesonet' 
    #img = img / 255.0
        
    return img, label

# Create data set -> train_dataset, val_dataset, test_dataset
def make_data(train_df, val_df, test_df, BATCH_SIZE):
    print(" Creating training dataset...")
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_df['filepath'].values, train_df['label'].values)
    )

    train_dataset = (
        train_dataset
        .map(load_image, num_parallel_calls=AUTOTUNE)       # Load images in parallel
        .filter(is_valid_image)                             # Remove corrupted images
        .map(preprocess_image, num_parallel_calls=AUTOTUNE) # Preprocess
        .shuffle(buffer_size=500 )                          # Shuffle for randomness      
        .batch(BATCH_SIZE)                                  # Create batches
        .prefetch(buffer_size=AUTOTUNE)                     # Prefetch next batch
    )

    print(" Creating validation dataset...")
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (val_df['filepath'].values, val_df['label'].values)
    )

    validation_dataset = (
        validation_dataset
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .filter(is_valid_image)
        .map(preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    print(" Creating test dataset...")
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_df['filepath'].values, test_df['label'].values)
    )

    test_dataset = (
        test_dataset
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .filter(is_valid_image)
        .map(preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    print(" All datasets created successfully!\n")

    return train_dataset, validation_dataset, test_dataset

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
        axes[idx].set_title(f"Label: {'Real' if label == 1 else 'Fake'}",
                        fontsize=12,
                        color='green' if label == 1 else 'red')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Sample Images from Training Set (After Preprocessing)',
                fontsize=14, y=1.02)
    plt.savefig(os.path.join(plot_dir, f'verify_pipeline_{file_dir}.png'), dpi=150, bbox_inches='tight')
    plt.close()