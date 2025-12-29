import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from keras import applications

# ==========================================
# CONFIGURATION
# ==========================================
# Path
MODEL_PATH = "Third_Year\DL_project\save\model\efficientnetb0_ffaugmented_20251203_064134.keras" 
INPUT_SIZE = (224, 224)

# ==========================================
# LOAD MODEL
# ==========================================
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    print("Make sure the path is correct and the file exists.")
    exit()

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================
def preprocess_image(image):
    """
    Same preprocessing as your training pipeline:
    1. Resize to 224x224
    2. EfficientNet preprocessing (normalization)
    """
    # Resize image 
    image = cv2.resize(image, INPUT_SIZE)
    
    # Convert to float32 and apply EfficientNet standard preprocessing
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)
    img_array = applications.efficientnet.preprocess_input(img_array)
    
    return img_array

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_deepfake(image):
    if image is None:
        return "Please upload an image."
    
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed_img, verbose=0)[0][0] # Sigmoid output (0.0 to 1.0)
    
    # Logic: 0 = Fake, 1 = Real (Based on your data_loader)
    # But usually datasets are 0=Real, 1=Fake. 
    # Let's check your data_loader.py...
    # Your data_loader says: labels.append(1 if label_name == 'real' else 0)
    # So: 1.0 is REAL, 0.0 is FAKE.
    
    confidence_real = float(prediction)
    confidence_fake = 1.0 - confidence_real
    
    # Return dictionary for Gradio Label output
    return {
        "Real": confidence_real, 
        "Fake": confidence_fake
    }

# ==========================================
# GRADIO INTERFACE
# ==========================================
# Create the interface
demo = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Confidence"),
    title="🕵️‍♂️ Deepfake Detection System",
    description=(
        "Upload a face image to check if it is **Real** or **Fake**.\n"
        "Model: EfficientNetB0 (Fine-tuned on FaceForensics++ c40)"
    ),
    examples=[
        # Add paths to sample images here if you have them
        # ["figures/real_sample.jpg"], 
        # ["figures/fake_sample.jpg"]
    ],
    theme="default"
)

# Launch the app
if __name__ == "__main__":
    print(" Launching Demo...")
    demo.launch(share=True)