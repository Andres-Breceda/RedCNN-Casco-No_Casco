import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Detector de Casco de Seguridad en Tiempo Real",
    page_icon="📸",
    layout="centered"
)

# --- CSS Personalizado para el Marco ---
st.markdown(
    """
    <style>
    .stApp {
        border: 1px solid #D3D3D3; /* Marco de 1px, sólido, color gris claro */
        border-radius: 5px; /* Bordes ligeramente redondeados (opcional) */
        padding: 20px; /* Espaciado interno para que el contenido no pegue con el marco */
        margin: 20px; /* Margen externo para que el marco no pegue con los bordes de la ventana */
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); /* Sombra suave (opcional) */
        background-color: whitesmoke; /* Color gris claro */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cargar modelo con cache
@st.cache_resource
def load_deep_learning_model():
    model = tf.keras.models.load_model("modelo_entrenado_completo.h5")
    return model

model = load_deep_learning_model()

#Encabezado de la pagina
st.header("Detector de Casco de Seguridad con Cámara en Vivo 👷 ")
st.write(
    """
    Esta aplicación fue realizada como Proyecto Final del Programa de Data Science.
    Te permite tomar una foto con tu cámara web para
    **detectar si una persona está usando un casco de seguridad**.
    """
)

st.markdown("---")

st.info("Sigue los siguientes pasos:")
st.info("1. Asegúrate de que la persona esté bien visible en el encuadre para una mejor predicción.")
st.info("2. Toma una Foto 👇")

# Widget de cámara
img_file_buffer = st.camera_input("Toma una foto para la predicción haciendo click en **Take Photo**")

if img_file_buffer is not None:
    # Convertir a imagen PIL
    bytes_data = img_file_buffer.getvalue()
    image = Image.open(io.BytesIO(bytes_data))

    st.image(image, caption="Imagen tomada")
    st.info("3. Si no te ha gustado puedes hacer click en **Clear Photo** y volver a tomarla...")
    st.info("4. Procesando tu Imagen... ⏳")

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

        st.info("5. Resultados de la Predicción:")
        st.success(f"La imagen fue clasificada como: **{predicted_class}** con una confianza de **{confidence*100:.2f}%**")

        st.write("Distribución de probabilidades:")
        st.write(f"• No_casco: {(1 - confidence)*100:.2f}%")
        st.write(f"• Casco: {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
else:
    st.info("Listo para comenzar?")
