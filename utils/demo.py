import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN 
from keras import applications

# ==========================================
# CONFIGURATION
# ==========================================
# UPDATE THIS PATH
MODEL_PATH = r"save/model/efficientnetb0_ffnormal_20250604.keras" 
INPUT_SIZE = (224, 224)

print("⏳ Initializing Face Detector...")
detector = MTCNN()

print(f"⏳ Loading Model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def preprocess_image(image):
    # Detect and Crop Face
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        image = image[y:y+h, x:x+w]
    
    image = cv2.resize(image, INPUT_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # NOTE: If using Xception, change this to applications.xception.preprocess_input
    img_array = applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_deepfake(image):
    if image is None: return "No Image"
    
    processed_img = preprocess_image(image)
    
    # Predict (Output 0.0 -> 1.0)
    # Since 1 = Fake, the prediction IS the Fake Confidence
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # === FIX: Updated Logic for 1=Fake ===
    confidence_fake = float(prediction)
    confidence_real = 1.0 - confidence_fake
    
    return {
        "Fake 🔴": confidence_fake,
        "Real 🟢": confidence_real
    }

demo = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Result"),
    title="🛡️ Deepfake Detection",
    description="Model: EfficientNetB0 | 0=Real, 1=Fake"
)

if __name__ == "__main__":
    demo.launch(share=True)