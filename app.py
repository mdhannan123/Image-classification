import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from tensorflow.keras.applications import VGG16

# Load the trained model
model = load_model('best_model.keras')

# Initialize the VGG16 model for generating bottleneck features
vgg16 = VGG16(include_top=False, weights='imagenet')

# Define necessary variables
img_width, img_height = 224, 224  # Typical size for VGG16

# Manually define class indices (Update this with your actual class indices)
class_indices = {'butterflies': 0, 'chickens': 1, 'elephants': 2, 'horses': 3, 'spiders': 4, 'squirells': 5}
classes = list(class_indices.keys())

# Function to classify new images
def classify_image(image_path, model):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image

    bottleneck_feature = vgg16.predict(img_array)
    prediction = model.predict(bottleneck_feature)
    
    max_confidence = np.max(prediction)
    class_index = np.argmax(prediction)
    class_label = classes[class_index]
    
    # Define confidence thresholds based on class labels
    if class_label in ['elephants', 'horses', 'squirrels']:
        confidence_threshold = 0.8  # Set threshold to 80% for specific classes
    else:
        confidence_threshold = 0.7  # Default threshold to 70% for other classes
    
    if max_confidence >= confidence_threshold:
        return class_label, max_confidence, prediction
    else:
        return "Unknown", max_confidence, prediction

# Streamlit UI
st.title('Image Classification using Trained Model')
st.write('Upload an image to classify')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with open("temp_image", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Classify the image
    class_label, confidence, prediction = classify_image("temp_image", model)

    if class_label == "Unknown":
        st.write(f'The model is not confident in predicting this image')
    else:
        st.write(f'Classified as: {class_label}')
