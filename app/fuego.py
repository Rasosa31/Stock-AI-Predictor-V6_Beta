import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # Ver TODO
import tensorflow as tf
import numpy as np

# 1. Configuración de ruta
path = "app/models_v6_beta/m1_puro.keras"
if not os.path.exists(path):
    path = "models_v6_beta/m1_puro.keras"

print(f"--- INICIANDO TEST DE FUERZA BRUTA ---")
print(f"Buscando modelo en: {path}")

try:
    print("1. Cargando modelo...")
    # Cargamos SIN optimización alguna para ver si es el cargador
    model = tf.keras.models.load_model(path, compile=False)
    print("✅ Modelo cargado.")

    print("2. Preparando datos sintéticos...")
    dummy_data = np.random.rand(1, 60, 9).astype(np.float32)

    print("3. Intentando PREDICCIÓN (Aquí es donde se congela la App)...")
    res = model.predict(dummy_data, verbose=1) 
    print(f"✅ ¡RESULTADO EXITOSO!: {res}")

except Exception as e:
    print(f"❌ ERROR DETECTADO: {e}")