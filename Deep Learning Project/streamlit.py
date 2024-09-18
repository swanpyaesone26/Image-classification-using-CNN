import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('gender_classification.h5')

# Function to make predictions
def predict_gender(image):
    # Preprocess the image (resize, normalize, etc.)
    # Then pass it through your model for prediction
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, size=(32, 32))  # Resize to match model input size
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize
    prediction = model.predict(img_array)
    return "Male" if prediction[0][0] > 0.5 else "Female"

# Streamlit app
def main():
    st.title("Gender Classification App")
    st.write("Upload an image to classify whether it's a male or female.")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction when button is clicked
        if st.button("Predict"):
            gender = predict_gender(image)
            st.write("Predicted Gender:", gender)

if __name__ == "__main__":
    main()


