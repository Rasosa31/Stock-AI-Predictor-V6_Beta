import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os
import json
from datetime import datetime
import pandas_ta as ta
import tensorflow as tf
import random

# --- REPRODUCIBILIDAD ---
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# --- CONFIGURACIÓN AMBIENTE ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(page_title="StockAI V6_Beta PRO V3.2", layout="wide")

# --- GESTIÓN DE SESGO (BIAS MEMORY) ---
BIAS_FILE = "bias_memory.json"

def load_bias():
    if os.path.exists(BIAS_FILE):
        try:
            with open(BIAS_FILE, "r") as f:
                return json.load(f)
        except: return {}
    return {}

def save_bias(ticker, error_percent, tf_key):
    bias_data = load_bias()
    composite_key = f"{ticker}_{tf_key}"
    current_bias = bias_data.get(composite_key, 0.0)
    alpha_map = {"Daily": 0.20, "Weekly": 0.10, "Monthly": 0.05}
    alpha = alpha_map.get(tf_key, 0.10)
    new_bias = (current_bias * (1 - alpha)) + (error_percent * alpha)
    bias_data[composite_key] = round(new_bias, 4)
    with open(BIAS_FILE, "w") as f:
        json.dump(bias_data, f)

def get_current_bias(ticker, tf_key):
    bias_data = load_bias()
    return bias_data.get(f"{ticker}_{tf_key}", 0.0)

# --- NUEVA FUNCIÓN: DETECCIÓN DE PATRONES DE VELAS ---
def get_candle_signals(df):
    if len(df) < 3: return 0, "Neutral"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body_last = abs(last['Close'] - last['Open'])
    
    score = 0
    pattern = "Sin Patrón"

    # 1. Bearish Engulfing (Bajista)
    if prev['Close'] > prev['Open'] and last['Close'] < last['Open'] and \
       last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']:
        score -= 2.5
        pattern = "⚠️ Envolvente Bajista"
    
    # 2. Bullish Engulfing (Alcista)
    elif prev['Close'] < prev['Open'] and last['Close'] > last['Open'] and \
         last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']:
        score += 2.0
        pattern = "✅ Envolvente Alcista"

    # 3. Shooting Star (Bajista)
    upper_shadow = last['High'] - max(last['Open'], last['Close'])
    if upper_shadow > (body_last * 2) and last['Close'] < last['Open']:
        score -= 1.5
        pattern = "☄️ Estrella Fugaz"

    # 4. Hammer (Alcista)
    lower_shadow = min(last['Open'], last['Close']) - last['Low']
    if lower_shadow > (body_last * 2) and last['Close'] > last['Open']:
        score += 1.5
        pattern = "🔨 Martillo"

    return score, pattern

# --- INICIALIZACIÓN DE ESTADOS ---
if 'resultados_escaneo' not in st.session_state:
    st.session_state.resultados_escaneo = None

if 'historial_consultas' not in st.session_state:
    st.session_state.historial_consultas = pd.DataFrame(columns=[
        "Fecha", "Activo", "TF", "Precio", "Predicción", "Dirección", "Potencial %", "Acuerdo %"
    ])

# --- CARGA DE MODELOS ---
MODELS_DIR = 'models_v6_beta'
@st.cache_resource
def get_v6_models():
    committee = {}
    model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]
    for name in model_names:
        p = os.path.join(MODELS_DIR, f"{name}.keras")
        if os.path.exists(p):
            try: committee[name] = tf.keras.models.load_model(p, compile=False)
            except: pass
    return committee

expertos_dict = get_v6_models()

# --- CORE FUNCTIONS ---
def get_data(ticker, timeframe):
    p_map = {"Daily": "10y", "Weekly": "max", "Monthly": "max"}
    i_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    try:
        df = yf.download(ticker, period=p_map[timeframe], interval=i_map[timeframe], progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).strip() for col in df.columns]
        if df.empty or len(df) < 100: return pd.DataFrame()
        df['SMA_100'] = df['Close'].rolling(100).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None: df['ADX'] = adx['ADX_14']
        return df.dropna().ffill().bfill()
    except: return pd.DataFrame()

def predict_ensemble_stable(df, strength):
    try:
        feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
        sc = RobustScaler()
        scaled = sc.fit_transform(df[feats].values)
        win = scaled[-60:].reshape(1, 60, len(feats)).astype(np.float32)
        
        weights = {"m1_puro": 0.30, "m3_tendencia": 0.25, "m2_volatilidad": 0.15, "m4_memoria": 0.15, "m5_agresivo": 0.15}
        preds, weighted_sum, total_weight = [], 0, 0
        
        for name, model in expertos_dict.items():
            p = model(win, training=False)
            val = float(p[0][0])
            preds.append(val)
            w = weights.get(name, 0.20)
            weighted_sum += val * w
            total_weight += w
        
        avg_pred = weighted_sum / total_weight
        curr_p = float(df['Close'].iloc[-1])
        vol = df['Close'].pct_change().std()
        z_score = (avg_pred - np.mean(scaled[:, 3])) / (np.std(scaled[:, 3]) + 1e-9)
        p_final = curr_p * (1 + (z_score * vol * strength))
        
        dispersion = (np.std(preds) / (np.mean(preds) + 1e-9)) * 100
        acuerdo = max(0, min(100, 100 - (dispersion * 50)))
        
        return p_final, acuerdo
    except: return None, None

def get_csv_download_link(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

def actualizar_memoria_errores():
    if not st.session_state.historial_consultas.empty:
        hist = st.session_state.historial_consultas
        hist_unique = hist.drop_duplicates(subset=['Activo', 'TF'], keep='first')
        for _, row in hist_unique.iterrows():
            df_now = get_data(row['Activo'], row['TF'])
            if not df_now.empty:
                real = df_now['Close'].iloc[-1]
                err = ((row['Predicción'] - real) / real) * 100
                save_bias(row['Activo'], err, row['TF'])
        st.sidebar.success("✅ Memoria de Sesgos Actualizada")

# --- INTERFAZ ---
st.title("🤖 Stock-AI Predictor V6_Beta PRO V3.2")

with st.sidebar:
    st.header("⚙️ Configuración")
    ticker_main = st.text_input("Símbolo Principal:", value="NQ=F").upper()
    tf_main = st.selectbox("Temporalidad:", ["Daily", "Weekly", "Monthly"])
    fuerza = st.slider("Sensibilidad de IA:", 0.1, 1.0, 0.4)
    agresividad_sesgo = st.slider("Impacto del Sesgo:", 1.0, 3.0, 1.5) # NUEVO: Control de impacto
    
    st.divider()
    if st.button("🔄 Sincronizar Sesgos (Fin de Jornada)"):
        actualizar_memoria_errores()
        st.rerun()
    
    current_b = get_current_bias(ticker_main, tf_main)
    st.info(f"Sesgo {tf_main} en {ticker_main}: {current_b:+.2f}%")

tab1, tab2, tab3 = st.tabs(["📊 Análisis Individual", "🧪 Backtesting Pro", "🚀 Escaneo Maestro"])

with tab1:
    df = get_data(ticker_main, tf_main)
    if not df.empty:
        df_f = df.tail(100)
        fig = go.Figure(data=[go.Candlestick(x=df_f.index, open=df_f['Open'], high=df_f['High'], low=df_f['Low'], close=df_f['Close'])])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Ejecutar Predicción Beta", key="run_main"):
            p_raw, a_val = predict_ensemble_stable(df, fuerza)
            c_score, c_pattern = get_candle_signals(df)
            
            if p_raw:
                # APLICAR SESGO CON MULTIPLICADOR DE IMPACTO
                p_final = p_raw * (1 - ((current_b * agresividad_sesgo) / 100))
                
                curr_p = float(df['Close'].iloc[-1])
                potencial = ((p_final - curr_p) / curr_p) * 100
                direccion = "🚀 COMPRA" if p_final > curr_p else "📉 VENTA"
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precio Actual", f"{curr_p:.2f}")
                c2.metric("Target Beta", f"{p_final:.2f}", f"{potencial:+.2f}%")
                c3.metric("Acción de Precio", c_pattern, f"Score: {c_score:+.1f}")
                c4.metric("Sesgo Aplicado", f"{(current_b * agresividad_sesgo):+.2f}%")
                
                st.markdown(f"**Recomendación:** {direccion}")
                
                nueva_fila = pd.DataFrame([{
                    "Fecha": datetime.now().strftime("%H:%M:%S"), "Activo": ticker_main,
                    "TF": tf_main, "Precio": round(curr_p, 2),
                    "Predicción": round(p_final, 2), "Dirección": direccion,
                    "Potencial %": round(potencial, 2), "Acuerdo %": round(a_val, 1)
                }])
                st.session_state.historial_consultas = pd.concat([nueva_fila, st.session_state.historial_consultas], ignore_index=True)

with tab3:
    st.subheader("🚀 Escaneo Maestro con Rating de Calidad")
    lista = st.text_area("Lista de Tickers:", value="AAPL, NVDA, BTC-USD, NQ=F, EURUSD=X", height=100)
    
    if st.button("🔍 Iniciar Escaneo Maestro"):
        tickers = [t.strip().upper() for t in lista.split(",") if t.strip()]
        results = []
        bar = st.progress(0)
        bias_map = load_bias()
        
        # --- DENTRO DEL BUCLE DEL ESCANEO MAESTRO (Tab 3) ---
        for idx, t in enumerate(tickers):
            df_t = get_data(t, tf_main)
            if not df_t.empty and len(df_t) >= 65:
                pf_raw, av = predict_ensemble_stable(df_t, fuerza)
                c_score, c_patt = get_candle_signals(df_t)
                
                if pf_raw:
                    t_bias = bias_map.get(f"{t}_{tf_main}", 0.0)
                    # Aplicar Sesgo con impacto
                    pf_corregido = pf_raw * (1 - ((t_bias * agresividad_sesgo) / 100))
                    cp = df_t['Close'].iloc[-1]
                    
                    # --- MEJORA: PRECISIÓN DINÁMICA (Especial para FX) ---
                    precision = 4 if cp < 10 else 2
                    
                    pot = ((pf_corregido - cp) / cp) * 100
                    
                    # --- LÓGICA DE RATING ---
                    abs_bias = abs(t_bias * agresividad_sesgo)
                    if av >= 75 and abs_bias <= 0.8 and c_score >= 0:
                        rating = "🏆 ORO"
                    elif (av >= 60 and abs_bias <= 1.5) or (c_score < 0 and pot > 0):
                        rating = "🥈 PLATA"
                    else:
                        rating = "🥉 BRONCE"
                    
                    if pot > 0 and c_score <= -2:
                        rating = "⚠️ DIVERGENCIA"

                    # --- ESTRUCTURA DE COLUMNAS COMPATIBLE (SEGÚN IMAGEN) ---
                    res_row = {
                        "Fecha": datetime.now().strftime("%H:%M"),
                        "Activo": t,
                        "TF": tf_main,
                        "Precio Cierre": round(cp, precision), # NOMBRE RESTAURADO
                        "Predicción": round(pf_corregido, precision), # NOMBRE RESTAURADO (ES EL TARGET BETA)
                        "Dirección": "🚀 COMPRA" if pf_corregido > cp else "📉 VENTA",
                        "Potencial %": round(pot, 2),
                        "Acuerdo %": round(av, 1),
                        "Sesgo %": round(t_bias, 2),
                        "Rating": rating,     # Extras al final para no romper comparativa
                        "Vela": c_patt,
                        "IA Pura": round(pf_raw, precision)
                    }
                    results.append(res_row)

                    # Sincronización de historial (Mantiene el mismo formato)
                    nf_h = pd.DataFrame([{
                        "Fecha": res_row["Fecha"], "Activo": t, "TF": tf_main, 
                        "Precio": round(cp, precision), "Predicción": round(pf_corregido, precision), 
                        "Dirección": res_row["Dirección"], "Potencial %": round(pot, 2), "Acuerdo %": av
                    }])
                    st.session_state.historial_consultas = pd.concat([st.session_state.historial_consultas, nf_h], ignore_index=True)

            bar.progress((idx+1)/len(tickers))
        
        if results:
            st.session_state.resultados_escaneo = pd.DataFrame(results).sort_values("Acuerdo %", ascending=False)

    if st.session_state.resultados_escaneo is not None:
        st.divider()
        st.dataframe(st.session_state.resultados_escaneo, use_container_width=True)
        col_d, col_l = st.columns([1, 4])
        col_d.download_button("📥 Descargar CSV", get_csv_download_link(st.session_state.resultados_escaneo), f"escaneo_beta_{datetime.now().strftime('%Y%m%d')}.csv")
        if col_l.button("🗑️ Limpiar Tabla"):
            st.session_state.resultados_escaneo = None
            st.rerun()