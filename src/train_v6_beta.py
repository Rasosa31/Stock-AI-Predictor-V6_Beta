import tensorflow as tf
tf.keras.backend.clear_session() # Limpia la memoria antes de empezar
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import os
import random

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# --- NUEVO: BLINDAJE DE REPRODUCIBILIDAD ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 1. CONFIGURACIÓN
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
PATH_DATA = "data/multi_stock_data.csv"
EPOCHS = 30 
BATCH_SIZE = 64
# Definimos 5 semillas diferentes para que el comité sea un "Ensemble" real
SEMILLAS_COMITE = [42, 7, 123, 888, 2026] 

def procesar_dataset_global(path):
    print(f"📖 Cargando dataset maestro: {path}")
    df_raw = pd.read_csv(path, index_col=0, parse_dates=True)
    X_global, y_global = [], []
    tickers = df_raw['Ticker'].unique()
    scaler = RobustScaler()

    for t in tickers:
        df_t = df_raw[df_raw['Ticker'] == t].copy()
        adx_df = ta.adx(df_t['High'], df_t['Low'], df_t['Close'], length=14)
        if adx_df is not None:
            df_t['ADX'] = adx_df['ADX_14']
        
        df_t.dropna(inplace=True)
        if len(df_t) < 100: continue
        
        scaled_t = scaler.fit_transform(df_t[FEATURES])
        for i in range(60, len(scaled_t)):
            X_global.append(scaled_t[i-60:i])
            y_global.append(scaled_t[i, 3]) 
            
    return np.array(X_global), np.array(y_global)

def crear_modelo_v6_beta(n_features):
    # Mantenemos tu arquitectura exitosa
    model = Sequential([
        Bidirectional(LSTM(70, return_sequences=True, input_shape=(60, n_features))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- EJECUCIÓN ---
if not os.path.exists(PATH_DATA):
    print(f"❌ Error: No se encuentra {PATH_DATA}.")
else:
    X, y = procesar_dataset_global(PATH_DATA)
    
    # Creamos carpeta nueva para no sobreescribir tu V6 de producción
    FOLDER_MODELS = 'models_v6_beta'
    if not os.path.exists(FOLDER_MODELS): 
        os.makedirs(FOLDER_MODELS)

    model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]

    callback_parada = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=5,
        restore_best_weights=True
    )

    # 2. Bucle de entrenamiento con SEMILLAS DIFERENTES
    for idx, name in enumerate(model_names):
        current_seed = SEMILLAS_COMITE[idx]
        print(f"\n🚀 Entrenando Experto: {name} | Semilla: {current_seed}")
        
        # Aplicamos la semilla ANTES de crear el modelo
        set_seeds(current_seed)
        
        model = crear_modelo_v6_beta(len(FEATURES))
        
        model.fit(
            X, y, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            verbose=1,
            callbacks=[callback_parada] 
        )
        
        # Guardamos con la nueva extensión y en la carpeta beta
        model.save(f'{FOLDER_MODELS}/{name}.keras')
        print(f"✅ {name} (Semilla {current_seed}) guardado en {FOLDER_MODELS}/")

    print("\n✨ ¡Comité V6-BETA (Ensemble de Semillas) completado!")