import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
# Page Configuration
st.set_page_config(page_title="Image Classification", layout="wide" ,page_icon="recycle-symbol.png")

# Sidebar Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("", ["Image Classifier",  "About"])


# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.hdf5')
    return model
model = load_model()


# Function to display the Heart Disease Prediction UI
def image_classifier():
    st.title("Image Classification: Organic vs Recyclable")
    
    st.write("##### Upload an image to determine if it's organic or recyclable.")
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Preprocess image
        img = image.resize((224,224))  # Adjust size if needed
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        # Make prediction
        prediction = model.predict(img_array)
        class_names=['Organic','Recycle']
        string="The image shown is  "+class_names[np.argmax(prediction)]+" Waste"
        st.success(string)
# Function to display About Section
def about_section():
    st.markdown("<h2 style='text-align: center;'>About This Project</h2>", unsafe_allow_html=True)
    st.write("""
    This is a **Deep Learning-based Image Classification System** built using **Streamlit**.
    - Users can classify the image if it is **Recyclable or Organic**.
    - Develop a CNN model to classify images of plastic waste into different categories
    - Built with **Python, Streamlit, and DL Models**.
    - Designed by **Rakesh R**.
    """)

# Route based on sidebar selection
if menu == "Image Classifier":
    image_classifier()

elif menu == "About":
    about_section()
