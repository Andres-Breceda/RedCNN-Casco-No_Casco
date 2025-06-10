import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf

# Cargar modelo con cache
@st.cache_resource
def load_deep_learning_model():
    model = tf.keras.models.load_model("modelo_entrenado_completo.h5")
    return model

model = load_deep_learning_model()

# Widget de cámara
img_file_buffer = st.camera_input("Toma una foto para la predicción")

if img_file_buffer is not None:
    # Convertir a imagen PIL
    bytes_data = img_file_buffer.getvalue()
    image = Image.open(io.BytesIO(bytes_data))

    st.image(image, caption="Imagen tomada", use_column_width=True)
    st.subheader("Procesando imagen para la predicción...")

    # Convertir a arreglo NumPy
    img_array = np.array(image)

    # Preprocesamiento
    try:
        img_resized = cv2.resize(img_array, (128, 128))
    except Exception as e:
        st.error(f"Error al redimensionar la imagen: {e}")
        st.stop()

    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # Añadir dimensión de lote

    # Predicción
    try:
        prediction = model.predict(img_input)

        class_names = ['No_casco', 'Casco']
        confidence = prediction[0][0]  # Valor entre 0 y 1
        predicted_class = class_names[1] if confidence >= 0.5 else class_names[0]

        st.subheader("Resultados de la Predicción:")
        st.success(f"La imagen fue clasificada como: **{predicted_class}** con una confianza de **{confidence*100:.2f}%**")

        st.write("Distribución de probabilidades:")
        st.write(f"• No_casco: {(1 - confidence)*100:.2f}%")
        st.write(f"• Casco: {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
else:
    st.info("Toma una foto con la cámara para comenzar.")