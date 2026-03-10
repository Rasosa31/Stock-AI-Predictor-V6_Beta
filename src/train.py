import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import tensorflow as tf
import random

# ==========================================
# 1. MEJORA V3: CONSISTENCIA (REPRODUCIBILIDAD)
# ==========================================
def set_seeds(seed=42):
    # 1. Variables de Entorno (Deben ir ANTES de cualquier operación de TF)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 2. Semillas de Librerías
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    # 3. Forzar un solo hilo (Opcional, pero garantiza 100% igualdad)
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

set_seeds(42)

def train_model():
    # 2. Carga de datos
    file_path = 'data/multi_stock_data.csv'
    if not os.path.exists(file_path):
        print(f"❌ Error: No se encuentra {file_path}. Ejecuta el descargador primero.")
        return

    df = pd.read_csv(file_path)
    
    # 3. ESTÁNDAR V3: 8 FEATURES
    # Asegúrate de que el CSV tenga estas columnas exactamente
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI']
    
    # Verificamos que todas las columnas existan en el CSV
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"❌ Error: Faltan columnas en el CSV: {missing}")
        return

    # Limpieza profunda de datos
    data_clean = df[features].bfill().ffill().dropna().values
    
    # 4. Escalar los datos (Matriz de 8 columnas)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_clean)
    
    # 5. Crear secuencias
    X, y = [], []
    window = 60
    
    # Buscamos el índice de 'Close' para que 'y' sea siempre el precio de cierre
    close_idx = features.index('Close')
    
    for i in range(window, len(scaled_data)):
        # X: Ventana de 60 días con las 8 columnas
        X.append(scaled_data[i-window:i, :]) 
        # y: El precio de Cierre del día actual
        y.append(scaled_data[i, close_idx])          
    
    X, y = np.array(X), np.array(y)
    
    # 6. Definición de la Red Neuronal V3
    # input_shape=(60, 8)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) # Predice 1 solo valor (el Cierre)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 7. Entrenamiento
    print(f"🚀 Iniciando Entrenamiento V3...")
    print(f"📊 Features: {len(features)} | Muestras: {len(X)}")
    
    model.fit(
        X, y, 
        batch_size=32, 
        epochs=15, # Un poco más de épocas para la V3
        shuffle=True, 
        verbose=1
    )
    
    # 8. PERSISTENCIA DE PESOS (Mejora V3)
    os.makedirs('models', exist_ok=True)
    
    # Guardamos el modelo en formato .h5 (o .keras)
    model.save('models/lstm_model.keras')
    
    # GUARDAR EL ESCALADOR ES VITAL: 
    # Sin este archivo, la App no podrá interpretar los dólares correctamente.
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ PROCESO COMPLETADO")
    print("📂 Modelo: models/lstm_model.keras")
    print("📂 Escalador: models/scaler.pkl")

if __name__ == "__main__":
    train_model()