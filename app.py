import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

st.write("Age and Gender Detection")


# Load the model
model_path = "D:/vg/agegender_resaved.h5"
model = tf.keras.models.load_model(model_path)


# Compile the model with the same loss functions and metrics
model.compile(
    loss=['binary_crossentropy', 'mean_absolute_error'],
    optimizer='adam',
    metrics={'gender_out': 'accuracy', 'age_out': 'mean_absolute_error'}
)


option = st.selectbox("OPTION", ("Upload Image", "Use Webcam"))
st.write("You selected:", option)

def process_image(img):
    pil_img = Image.open(img)
    pil_img = ImageOps.grayscale(pil_img)
    resized_img = pil_img.resize((128, 128))
    img_array = np.array(resized_img)
    img_array = img_array / 255.0
    input_img = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    input_img = np.expand_dims(input_img, axis=0)   # Add batch dimension
    return input_img

def get_predictions(input_img):
    try:
        gender_prediction, age_prediction = model.predict(input_img)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None
    gender_label = "Male" if gender_prediction < 0.5 else "Female"
    age_label = int(round(age_prediction[0][0] * 100))  # Denormalize age values back to the range [0, 100]
    age_label = max(0, min(100, age_label))
    return gender_label, age_label

def get_age_category(age):
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teenager"
    elif age <= 39:
        return "Young Adult"
    elif age <= 59:
        return "Adult"
    else:
        return "Elderly"

if option == "Upload Image":
    img = st.file_uploader(label="Load Image", type=["png", "jpg", "jpeg"])
    if img is not None:
        st.write("Image Uploaded Successfully.")  # Add this line for debugging
        input_img = process_image(img)
        st.write("Processed image shape:", input_img.shape)  # Add this line for debugging
        gender_label, age_label = get_predictions(input_img)
        st.write("Gender Label:", gender_label)  # Add this line for debugging
        st.write("Age Label:", age_label)  # Add this line for debugging
        if gender_label is not None and age_label is not None:
            st.write("Predicted Gender:", gender_label)
            st.write("Predicted Age:", age_label)
            age_category = get_age_category(age_label)
            st.write("Predicted Age Category:", age_category)
            pil_img = Image.open(img)
            resized_img = pil_img.resize((128, 128))
            st.image(resized_img, caption='Uploaded Image', use_column_width=True)

elif option == "Use Webcam":
    st.write("Press 'Start' to begin webcam feed and 'Stop' to end it.")
    start_button = st.button("Start")
    stop_button = st.button("Stop")

    if start_button:
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (128, 128))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=-1)
            input_frame = np.expand_dims(input_frame, axis=0)
            gender_label, age_label = get_predictions(input_frame)

            if gender_label and age_label:
                cv2.putText(frame, f"Gender: {gender_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Age: {age_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                FRAME_WINDOW.image(frame)
            
            if stop_button:
                cap.release()
                break
        cv2.destroyAllWindows()
