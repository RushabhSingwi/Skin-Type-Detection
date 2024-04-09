import traceback

import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

IMAGE_SHAPE = (224, 224)
classes = ["Dry Skin", "Oily Skin"]

st.title("Mirror.AI")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("RealTimeDetections.h5")
        return model, None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, str(e)

model, error_message = load_model()

if model is None:
    st.error("Error loading the model. Please check the model file.")
    st.stop()

with st.spinner('Loading model into memmory...'):
    model = load_model()


def load_and_prep_image(image):
    """
    Reads an image from filename, turns it into a tensor and reshapes
    it to (img_shape, img_shape,, color_channels)
    """
    # Read in the image
    # img = tf.io.read_file(filename)
    # Decode the read file into a tensor
    image = tf.image.decode_image(image)
    # Resize the image
    image = tf.image.resize(image, size=IMAGE_SHAPE)
    #Grayscale
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
        # Rescale the image (getting all values between 0 & 1)
        # image = image/255

    return image

def url_uploader():
    st.text("Provide URL for your image of skin")

    path = st.text_input("Enter image URL to classify...", "https://i.ibb.co/Tg62Zjv/oily-6.jpg")
    if path:
        try:
            content = requests.get(path).content
            st.write("Predicted Skin type :")
            with st.spinner("Classifying....."):
                img = load_and_prep_image(content)
                if img is not None:
                    model, error_message = load_model()
                    if model is not None:
                        label = model.predict(tf.expand_dims(img, axis=0))
                        st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
                        image = Image.open(BytesIO(content))
                        st.image(image, caption="Classifying the Skin", use_column_width=True)
                    else:
                        st.error("Error loading the model: {}".format(error_message))
        except Exception as e:
            st.error("Error processing image from URL: {}".format(e))

def file_Uploader():
    file = st.file_uploader("Upload file", type=["png", "jpeg", "jpg"])
    if not file:
        st.info("Upload a picture of the skin you want to predict.")
        return

    content = file.getvalue()
    model, error_message = load_model()  # Load the model
    if model is None:
        st.error("Error loading the model: {}".format(error_message))
        return

    st.write("Predicted Skin type :")
    with st.spinner("Classifying....."):
        img = load_and_prep_image(content)
        if img is not None:
            label = model.predict(tf.expand_dims(img, axis=0))
            st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption="Classifying the Skin", use_column_width=True)



file_Uploader()

#if function == 'URL':
#    url_uploader()
#else :
#    file_Uploader()