import streamlit as st
from PIL import Image
import io
from tensorflow.keras.models import load_model
from io import StringIO
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Cargar el modelo una vez al iniciar la app
model = load_model('modelo_entrenado.keras')

# Definir los estilos CSS para mejorar la apariencia
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Aplicar estilos CSS
local_css("style.css")

# Ruta de la carpeta principal
ruta_carpeta = 'archive/test'

# Lista para almacenar los nombres de las subcarpetas
labels = []

# Recorremos los elementos en el directorio dado
for elemento in os.listdir(ruta_carpeta):
    # Construimos la ruta completa del elemento
    ruta_completa = os.path.join(ruta_carpeta, elemento)
    # Verificamos si el elemento es una carpeta
    if os.path.isdir(ruta_completa):
        labels.append(elemento)


def set_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f4f4f2;
        }
        .css-18e3th9 {
            background-color: #f4f4f2;
        }
        h1 {
            color: #008080;
        }
        .st-bx {
            border-color: #008080;
        }
        .st-bv {
            color: #008080;
        }
        .st-bw {
            color: #008080;
        }
        </style>
        """, unsafe_allow_html=True)

def get_model_summary():
    """Obtiene el resumen del modelo como cadena de texto."""
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def convert_to_jpg(img):
    """Convierte una imagen a formato JPEG si no está ya en ese formato."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def load_and_process_image(uploaded_file):
    """Carga y preprocesa la imagen para el modelo."""
    img = Image.open(uploaded_file)
    if not uploaded_file.name.lower().endswith('.jpg'):
        img = convert_to_jpg(img)
    img = img.resize((224, 224))  # Asumiendo que tu modelo espera imágenes de 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar a [0, 1]
    return img_array

def plot_layer_weights(layer_index):
    """Plot the weights of a specific layer."""
    weights = model.layers[layer_index].get_weights()[0]  # Obtener los pesos
    if weights.ndim > 1:  # Solo plotea si los pesos son de más de 1 dimensión
        plt.figure(figsize=(10, 5))
        plt.imshow(weights[:, :, 0, 0], aspect='auto', cmap='viridis')  # Ajustar para diferentes tipos de capas
        plt.colorbar()
        plt.title(f'Pesos de la capa {layer_index}')
        plt.xlabel('Kernel Size')
        plt.ylabel('Filters')
        st.pyplot(plt)
    else:
        st.text("La capa seleccionada no tiene pesos visibles.")

def main():

    st.title("Reconocimiento de Imágenes de Aves")

    # Menú lateral
    menu = st.sidebar.selectbox("Menu", ["Predecir Imagen", "Ver Modelo", "Ver Pesos de Capa"])

    if menu == "Ver Modelo":
        st.text("Resumen del Modelo:")
        model_summary = get_model_summary()
        st.text(model_summary)
    elif menu == "Cargar Imagen":
        uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = load_and_process_image(uploaded_file)
            st.image(uploaded_file, caption='Imagen Cargada', use_column_width=True)
            st.success("Imagen cargada y convertida a JPG correctamente si fue necesario.")
    elif menu == "Ver Pesos de Capa":
        layer_index = st.sidebar.number_input("Indica el índice de la capa", min_value=1, max_value=len(model.layers)-1, step=1, value=1)
        plot_layer_weights(layer_index)
    elif menu == "Predecir Imagen":
        uploaded_file = st.file_uploader("Sube una imagen para predecir", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            img_array = load_and_process_image(uploaded_file)
            st.image(uploaded_file, caption='Imagen a Predecir', use_column_width=True)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            class_probability = predictions[0][predicted_class]

            # Mostrar la clase predicha y la probabilidad
            st.write(f'Clase Predicha: {labels[predicted_class]}')
            st.write(f'Probabilidad: {class_probability:.3%}')  # Formatea como porcentaje

if __name__ == "__main__":
    main()
