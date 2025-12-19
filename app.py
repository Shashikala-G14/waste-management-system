import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained Keras model
# Ensure the model file 'Waste Classification Model.h5' is in the same directory or provide the full path.
model = load_model('Waste Management System.h5')

# Define the class names for prediction output
class_names = ['Organic', 'Recyclable']

# Function to preprocess the image and make a prediction
def predict_waste_type(image_data):
  
    # Resize the image to the target size expected by the model
    img = image_data.resize((224, 224))
    img = img.convert("RGB")
    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the image (if the model was trained with normalized data)
    # Assuming the model was trained with images scaled to 1./255
    img_array = img_array / 255.0

    # Expand dimensions to match the model's input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)

    # The model outputs probabilities. For binary classification with sigmoid, if there's one output neuron
    # If the model has 2 output neurons with softmax and categorical_crossentropy, it will output 2 probabilities.
    # Let's assume binary_crossentropy with sigmoid (1 output neuron) or softmax with 2 output neurons.
    # For binary_crossentropy with sigmoid, predictions will be a single value between 0 and 1.
    # For categorical_crossentropy with 2 output neurons, predictions will be an array like [prob_organic, prob_recyclable]

    # Based on the model.add(Activation("sigmoid")) in cell 238ea04b and loss="binary_crossentropy",
    # it implies a single output neuron for binary classification.
    # A prediction value closer to 0 might indicate one class and closer to 1 the other.
    # The original code's predict_func used np.argmax for output, implying a 2-output neuron setup
    # but with sigmoid, it's usually a single output for one class probability. Let's adjust.

    # Given the previous context: `if result == 0: print("This image -> Recyclable") elif result == 1: print("This image -> Organic")`
    # This suggests that class 0 is 'Recyclable' and class 1 is 'Organic'.

    # With `model.add(Activation("sigmoid"))` for `numberOfClass` (which is 2)
    # and `loss="binary_crossentropy"`, this is a bit ambiguous.
    # If it's a single output neuron (binary classification), predictions is a (1,1) array.
    # If it's two output neurons (multi-class with binary loss), predictions is a (1,2) array.
    # Let's assume it's set up for two output neurons based on `Dense(numberOfClass)` and `np.argmax` in `predict_func`.

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    return predicted_class

# Streamlit app layout
st.title("Waste Classification App")
st.write("Upload an image to classify it as Organic or Recyclable waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Make prediction on button click
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            predicted_label = predict_waste_type(image)
            st.success(f"Prediction: This waste is **{predicted_label}**")

st.markdown("--- App Developed for Waste Classification --- ")

# To run this Streamlit app:
# 1. Save this code into a Python file (e.g., `app.py`).
# 2. Open your terminal or command prompt.
# 3. Navigate to the directory where you saved `app.py`.
# 4. Run the command: `streamlit run app.py`
# 5. The app will open in your web browser.