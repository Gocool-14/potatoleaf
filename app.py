import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Load model (only once at start)
model = load_model("potato_model.keras")

# ✅ Class labels (same order as training)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# UI
st.title("🌿 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width='stretch')

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.subheader(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence * 100:.2f}%")