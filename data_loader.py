# ==========================================
# LOAD DATA
# ==========================================

# Import modules
from sklearn.model_selection import train_test_split
import pandas as pd
import os

SEED = 42
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Scan directory --> filepath, labels
def scan(DATA_DIR):
    print("Scanning for images...")
    filepaths = []
    labels = []

    # Loop through 'fake' and 'real' folders
    for label_name in ['fake', 'real']:
        class_dir = os.path.join(DATA_DIR, label_name)

        # Check if directory exists
        if not os.path.isdir(class_dir):
            print(f" Warning: Directory not found - {class_dir}")
            continue

        # Walk through directory and subdirectories
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(dirpath, fname)
                    filepaths.append(full_path)
                    labels.append(1 if label_name == 'real' else 0)

        print(f" Found {labels.count(1 if label_name == 'real' else 0)} images in '{label_name}' folder")
    return filepaths, labels

# Create DataFrame --> df
def make_df(filepaths, labels):
    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\n Total samples: {len(df)}")
    print(f"   - Fake (0): {(df['label'] == 0).sum()}")
    print(f"   - Real (1): {(df['label'] == 1).sum()}")
    
    return df

# Check class imbalance
def check_imbalance(df):
    fake_count = (df['label'] == 0).sum()
    real_count = (df['label'] == 1).sum()
    balance_ratio = min(fake_count, real_count) / max(fake_count, real_count)
    print(f"\n  Class balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")

    if balance_ratio < 0.8:
        print("     Dataset is imbalanced! Consider using class weights during training.")

# Split into train/val/test --> train_df, val_df, test_df
def split_data(df): 
    print("\n Splitting dataset...")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df['label']  # Ensures same ratio in all splits
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df['label']
    )

    # Verify splits maintain class balance
    print("\n Class distribution in splits:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        fake = (split_df['label'] == 0).sum()
        real = (split_df['label'] == 1).sum()
        print(f"   {name:10s} - Fake: {fake:5d} | Real: {real:5d}")

    print("\n DataFrame created successfully!")
    
    return train_df, val_df, test_df