import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Import utils
# Đảm bảo bạn đang để file này cùng cấp với thư mục utils
from utils import data_loader, preprocessing, main_eval, save

# ==========================================
# ⚙️ CẤU HÌNH (BẠN CẦN SỬA 2 DÒNG NÀY)
# ==========================================

# 1. 👇 Dán đường dẫn file model .keras của bạn vào đây
# Ví dụ: "save/model/efficientnetb0_ffnormal_20251230_155300.keras"
MODEL_PATH_TO_LOAD = "save/model/TÊN_FILE_MODEL_CỦA_BẠN.keras" 

# 2. 👇 Dán đường dẫn file history .npy của bạn vào đây
# Ví dụ: "save/history/efficientnetb0_ffnormal_20251230_155300.npy"
HISTORY_PATH_TO_LOAD = "save/history/TÊN_FILE_HISTORY_CỦA_BẠN.npy"

# ------------------------------------------
# CÁC THÔNG SỐ KHÁC (GIỮ NGUYÊN NHƯ LÚC TRAIN)
# ------------------------------------------
MODEL_NAME   = "efficientnetb0" 
DATA_DIR     = "/mnt/d/PROJECT/virtual_env/DL/Project/FINAL_DATASET/Normal_Dataset/" 
DATASET_NAME = 'ffnormal'
BATCH_SIZE   = 32
IMAGE_SIZE   = (224, 224)
DROPOUT_RATE = 0.2  # 0.5 nếu là mesonet
LEARNING_RATE = 0.001 

# Setup đường dẫn save (Tự động tạo tên file mới có đuôi _RESUME)
SAVE_DIR = "save"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
file_dir     = f"{MODEL_NAME}_{DATASET_NAME}_{current_time}_RESUME"

# Tạo folder output nếu chưa có
for folder in ['plot', 'test']:
    os.makedirs(os.path.join(SAVE_DIR, folder), exist_ok=True)

# ==========================================
# 1. LOAD LẠI DỮ LIỆU & MODEL
# ==========================================
print("\n" + "="*40)
print(" 1. LOAD DATA & MODEL")
print("="*40)

print(f"📂 Đang đọc dữ liệu từ: {DATA_DIR}")
train_df, val_df, test_df = data_loader.load_datasets(DATA_DIR)

# Gộp DataFrame để tính thống kê cho hàm save_metrics
df = pd.concat([train_df, val_df, test_df])

print("⚙️ Đang xử lý dữ liệu (Preprocessing)...")
# Chỉ cần tạo test_ds để đánh giá
# Lưu ý: train_ds và val_ds có thể bỏ qua để tiết kiệm RAM nếu không cần dùng lại
_, _, test_ds = preprocessing.make_data(train_df, val_df, test_df, BATCH_SIZE, MODEL_NAME)

print(f"🧠 Đang load model từ: {MODEL_PATH_TO_LOAD}")
try:
    model = tf.keras.models.load_model(MODEL_PATH_TO_LOAD)
    print("✅ Load model thành công!")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    print("👉 Hãy kiểm tra lại đường dẫn trong biến MODEL_PATH_TO_LOAD")
    exit()

# ==========================================
# 2. XỬ LÝ HISTORY (LỊCH SỬ HUẤN LUYỆN)
# ==========================================
print("\n" + "="*40)
print(" 2. CHECKING TRAINING HISTORY")
print("="*40)

history = {}

# Thử load file history thật
if HISTORY_PATH_TO_LOAD and os.path.exists(HISTORY_PATH_TO_LOAD):
    print(f"📈 Tìm thấy file history: {HISTORY_PATH_TO_LOAD}")
    try:
        history = np.load(HISTORY_PATH_TO_LOAD, allow_pickle=True).item()
        print("✅ Load history thành công. Đang vẽ lại biểu đồ Training...")
        
        # Vẽ lại biểu đồ Loss/AUC/Accuracy
        main_eval.plot_history(history, f"{SAVE_DIR}/plot", file_dir)
    except Exception as e:
        print(f"⚠️ File history bị lỗi hoặc không đọc được: {e}")
        print("➡️ Sẽ sử dụng history rỗng (dummy) để chạy tiếp.")
else:
    print("⚠️ Không tìm thấy đường dẫn file history.")
    print("➡️ Sẽ sử dụng history rỗng (dummy) để chạy tiếp.")

# Đảm bảo history có cấu trúc đúng để hàm save không bị lỗi
required_keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'auc', 'val_auc']
for key in required_keys:
    if key not in history:
        history[key] = [] 

# ==========================================
# 3. CHẠY ĐÁNH GIÁ (TEST EVALUATE)
# ==========================================
print("\n" + "="*40)
print(" 3. RUNNING EVALUATION ON TEST SET")
print("="*40)

# Gọi hàm test_evaluate (Phiên bản mới trả về 11 giá trị)
results, tn, fp, fn, tp, fpr, fnr, f1_score, y_true, y_pred, y_pred_probs = main_eval.test_evaluate(
    model, test_ds, f"{SAVE_DIR}/plot", file_dir, f"{SAVE_DIR}/test"
)

# ==========================================
# 4. LƯU KẾT QUẢ (SAVE METRICS & PREDICTIONS)
# ==========================================
print("\n" + "="*40)
print(" 4. SAVING RESULTS")
print("="*40)

print("💾 Đang lưu file metrics (JSON)...")
save.save_metrics(
    MODEL_NAME, DATASET_NAME, current_time, 
    results, f1_score, tn, fp, fn, tp, fpr, fnr, 
    history, BATCH_SIZE, LEARNING_RATE, IMAGE_SIZE, DROPOUT_RATE, 
    df, train_df, val_df, test_df, 
    f"{SAVE_DIR}/test", file_dir
)

print("💾 Đang lưu file dự đoán (NPZ)...")
save.save_pred(
    f"{SAVE_DIR}/test", file_dir, f"{SAVE_DIR}/plot", 
    y_true, y_pred, y_pred_probs, results
)

print("\n🎉 HOÀN TẤT! Đã chạy xong toàn bộ phần đánh giá.")
print(f"👉 Kiểm tra kết quả tại thư mục: {SAVE_DIR}/test/ và {SAVE_DIR}/plot/")