# ==========================================
# LOAD DATA FROM SPLIT FOLDERS
# ==========================================
import pandas as pd
import os

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def scan_folder(folder_path):
    filepaths = []
    labels = []

    if not os.path.exists(folder_path):
        print(f" Warning: Folder not found: {folder_path}")
        return pd.DataFrame()

    for label_name in ['fake', 'real']:
        class_dir = os.path.join(folder_path, label_name)
        if not os.path.isdir(class_dir):
            continue

        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    filepaths.append(os.path.join(dirpath, fname))
                    labels.append(0 if label_name == 'real' else 1) # Labelling
    
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

def load_datasets(DATA_DIR):
    print(" Loading datasets from folders...")
    
    train_df = scan_folder(os.path.join(DATA_DIR, 'train'))
    val_df   = scan_folder(os.path.join(DATA_DIR, 'val'))
    test_df  = scan_folder(os.path.join(DATA_DIR, 'test'))

    print(f"\n Dataset Statistics:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    return train_df, val_df, test_df

def check_imbalance(df, name="Dataset"):
    #0 is real and 1 is fake
    fake = (df['label'] == 1).sum()
    real = (df['label'] == 0).sum()
    print(f"  {name}: Fake={fake}, Real={real} (Ratio: {min(fake,real)/max(fake,real):.2f})")