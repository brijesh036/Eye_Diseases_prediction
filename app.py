import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from io import BytesIO
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = "ODIR_VGG16_1.h5"  # Ensure this is in the same directory or provide the correct path
model = load_model(MODEL_PATH)

# Disease classes
disease_classes = ["Normal", "Diabetes", "Glaucoma", "Hypertension", "Myopia", "Age Issues", "Other", "Cataract"]

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image to match the model's input requirements."""
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Streamlit app
st.title("🩺 Eye Disease Prediction with Explainable AI")
st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose a fundus image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("---")

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        st.write("Preprocessed Image Shape:", preprocessed_image.shape)

        # Model Prediction
        prediction = model.predict(preprocessed_image)
        

        predicted_class_index = np.argmax(prediction)
        predicted_class = disease_classes[predicted_class_index]
        confidence = np.max(prediction) * 100

        st.subheader("🎯 Prediction Result")
        st.markdown(f"**Predicted Disease:** {predicted_class}")
        #st.markdown(f"**Confidence Level:** {confidence:.2f}%")
        st.markdown("---")

        # LIME Explanation
        st.subheader("🔍 LIME Explanation")
        explainer = lime_image.LimeImageExplainer()

        # Convert the image back to uint8 for LIME
        lime_input_image = (preprocessed_image[0] * 255).astype(np.uint8)

        explanation = explainer.explain_instance(
            lime_input_image,
            lambda x: model.predict(x / 255.0),  # Normalize inside LIME predict function
            top_labels=1,
            hide_color=0,
            num_samples=1000  # Increased samples for more robust explanations
        )
        temp, mask = explanation.get_image_and_mask(
            label=predicted_class_index,
            positive_only=True,
            hide_rest=False,
            num_features=5
        )

        lime_fig = plt.figure(figsize=(8, 8))
        plt.imshow(mark_boundaries(lime_input_image, mask))
        plt.title("LIME Explanation")
        plt.axis("off")

        lime_buf = BytesIO()
        lime_fig.savefig(lime_buf, format="png")
        lime_buf.seek(0)
        st.image(lime_buf, caption="LIME Explanation", use_container_width=True)
        st.markdown("---")
        st.success("Analysis Complete! Explore the insights above.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a fundus image to get started.")

