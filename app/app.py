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

# --- MEJORA V3: REPRODUCIBILIDAD ---
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# --- CONFIGURACIÓN AMBIENTE ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(page_title="StockAI V6_Beta PRO", layout="wide")

# --- GESTIÓN DE SESGO (BIAS MEMORY) ---
BIAS_FILE = "bias_memory.json"

# --- BUSCA Y REEMPLAZA ESTAS FUNCIONES ---
def load_bias():
    if os.path.exists(BIAS_FILE):
        try:
            with open(BIAS_FILE, "r") as f:
                return json.load(f)
        except: return {}
    return {}

def save_bias(ticker, error_percent, tf_key):
    bias_data = load_bias()
    composite_key = f"{ticker}_{tf_key}" # LA CLAVE DEL ÉXITO: Ticker + Marco Temporal
    
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

def actualizar_memoria_errores():
    """Sincroniza el historial de predicciones con los cierres reales"""
    if not st.session_state.historial_consultas.empty:
        # Quitamos duplicados para no procesar el mismo par Ticker_TF varias veces
        hist = st.session_state.historial_consultas.drop_duplicates(subset=['Activo', 'TF'], keep='first')
        
        for _, row in hist.iterrows():
            df_now = get_data(row['Activo'], row['TF'])
            if not df_now.empty:
                real = df_now['Close'].iloc[-1]
                # Comparamos la predicción guardada vs el cierre actual de Yahoo Finance
                err = ((row['Predicción'] - real) / real) * 100
                save_bias(row['Activo'], err, row['TF'])
        st.sidebar.success(f"✅ Memoria D/W/M actualizada")

# --- INICIALIZACIÓN DE MEMORIA (HISTORIAL) ---
if 'historial_consultas' not in st.session_state:
    st.session_state.historial_consultas = pd.DataFrame(columns=[
        "Fecha", "Activo", "TF", "Precio", "Predicción", "Dirección", "Potencial %", "Acuerdo %"
    ])

# --- CARGA DE MODELOS ---
MODELS_DIR = 'models_v6_beta'
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join('app', 'models_v6_beta')

@st.cache_resource
def get_v6_models():
    committee = {}
    model_names = ["m1_puro", "m2_volatilidad", "m3_tendencia", "m4_memoria", "m5_agresivo"]
    for name in model_names:
        p = os.path.join(MODELS_DIR, f"{name}.keras")
        if os.path.exists(p):
            try:
                committee[name] = tf.keras.models.load_model(p, compile=False)
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
        
        # Nueva métrica de acuerdo mejorada
        dispersion = (np.std(preds) / (np.mean(preds) + 1e-9)) * 100
        acuerdo = max(0, min(100, 100 - (dispersion * 50)))
        
        return p_final, acuerdo
    except: return None, None

def get_csv_download_link(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

def actualizar_memoria_errores():
    if not st.session_state.historial_consultas.empty:
        hist = st.session_state.historial_consultas
        for _, row in hist.iterrows():
            df_now = get_data(row['Activo'], row['TF'])
            if not df_now.empty:
                real = df_now['Close'].iloc[-1]
                err = ((row['Predicción'] - real) / real) * 100
                save_bias(row['Activo'], err, row['TF'])
        st.sidebar.success("✅ Sesgos actualizados")

# --- INTERFAZ ---
st.title("🤖 Stock-AI Predictor V6_Beta PRO")

with st.sidebar:
    st.header("⚙️ Configuración")
    ticker_main = st.text_input("Símbolo Principal:", value="NQ=F").upper()
    tf_main = st.selectbox("Temporalidad:", ["Daily", "Weekly", "Monthly"])
    fuerza = st.slider("Sensibilidad:", 0.1, 1.0, 0.4)
    
    st.divider()
    st.subheader("🧠 Aprendizaje V3")
    if st.button("🔄 Sincronizar Sesgos (Ayer/Hoy)"):
        actualizar_memoria_errores()
        st.rerun()
    
    bias_map = load_bias()
    current_b = get_current_bias(ticker_main, tf_main)
    st.info(f"Sesgo en {tf_main} en {ticker_main}: {current_b:+.2f}%")
    st.write(f"Modelos Activos: {len(expertos_dict)}/5")

tab1, tab2, tab3 = st.tabs(["📊 Análisis Individual", "🧪 Backtesting Pro", "🚀 Escaneo Maestro"])

with tab1:
    df = get_data(ticker_main, tf_main)
    if not df.empty:
        df_f = df.tail(150)
        fig = go.Figure(data=[go.Candlestick(x=df_f.index, open=df_f['Open'], high=df_f['High'], low=df_f['Low'], close=df_f['Close'])])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Ejecutar Predicción Beta", key="run_main"):
            p_raw, a_val = predict_ensemble_stable(df, fuerza)
            if p_raw:
                # APLICAR CAPA DE CORRECCIÓN RESIDUAL
                p_final = p_raw * (1 - (current_b / 100))
                
                curr_p = float(df['Close'].iloc[-1])
                potencial = ((p_final - curr_p) / curr_p) * 100
                direccion = "🚀 COMPRA" if p_final > curr_p else "📉 VENTA"
                precision = 4 if curr_p < 20 else 2
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precio Actual", f"{curr_p:.{precision}f}")
                c2.metric("Target Corregido", f"{p_final:.{precision}f}", f"{potencial:+.2f}%")
                c3.metric("Acuerdo Beta", f"{a_val:.1f}%")
                c4.metric("Sesgo Aplicado", f"{current_b:+.2f}%", "🎯" if abs(current_b)<0.5 else "⚠️")
                
                st.markdown(f"**Dirección Sugerida:** {direccion}")
                
                nueva_fila = pd.DataFrame([{
                    "Fecha": datetime.now().strftime("%H:%M:%S"), "Activo": ticker_main,
                    "TF": tf_main, "Precio": round(curr_p, precision),
                    "Predicción": round(p_final, precision), "Dirección": direccion,
                    "Potencial %": round(potencial, 2), "Acuerdo %": round(a_val, 1)
                }])
                st.session_state.historial_consultas = pd.concat([nueva_fila, st.session_state.historial_consultas], ignore_index=True)

        if not st.session_state.historial_consultas.empty:
            st.divider()
            st.subheader("📋 Historial de Sesión")
            st.dataframe(st.session_state.historial_consultas, use_container_width=True)
            col_s, col_c = st.columns([1, 4])
            col_s.download_button("📥 Guardar CSV", get_csv_download_link(st.session_state.historial_consultas), f"consultas_{datetime.now().strftime('%H%M')}.csv", "text/csv")
            if col_c.button("🗑️ Limpiar"):
                st.session_state.historial_consultas = pd.DataFrame(columns=st.session_state.historial_consultas.columns)
                st.rerun()

with tab2:
    st.subheader("🧪 Simulación Histórica (m1_puro)")
    if not df.empty:
        test_days = st.number_input("Velas de prueba:", 5, 100, 20)
        if st.button("📊 Correr Backtest", key="run_bt"):
            hits, log = [], []
            prog = st.progress(0)
            feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_100', 'SMA_200', 'RSI', 'ADX']
            sc = RobustScaler()
            scaled = sc.fit_transform(df[feats].values)
            model_bt = expertos_dict.get("m1_puro")
            if model_bt:
                rango = range(len(scaled) - test_days, len(scaled))
                for idx, i in enumerate(rango):
                    win = scaled[i-60:i].reshape(1, 60, len(feats)).astype(np.float32)
                    p_val = float(model_bt(win, training=False)[0][0])
                    p_dir = "ALZA" if p_val > scaled[i-1, 3] else "BAJA"
                    r_dir = "ALZA" if scaled[i, 3] > scaled[i-1, 3] else "BAJA"
                    hit = 1 if p_dir == r_dir else 0
                    hits.append(hit)
                    log.append({"Fecha": df.index[i].strftime("%Y-%m-%d"), "Pred": p_dir, "Real": r_dir, "Hit": "✅" if hit else "❌"})
                    prog.progress((idx+1)/len(rango))
                st.success(f"Efectividad: {(sum(hits)/len(hits))*100:.1f}%")
                st.dataframe(pd.DataFrame(log), use_container_width=True)

with tab3:
    st.subheader("🚀 Escaneo Maestro (347 Activos)")
    lista = st.text_area("Lista de Tickers:", value="AAPL, NVDA, BTC-USD, NQ=F, EURUSD=X", height=100)
    if st.button("🔍 Iniciar Escaneo"):
        tickers = [t.strip().upper() for t in lista.split(",") if t.strip()]
        results = []
        bar = st.progress(0)
        
        # Cargamos sesgos una sola vez para velocidad
        bias_map = load_bias()
        
        for idx, t in enumerate(tickers):
            df_t = get_data(t, tf_main)
            if not df_t.empty and len(df_t) >= 65:
                pf_raw, av = predict_ensemble_stable(df_t, fuerza)
                if pf_raw:
                    # Buscamos el sesgo específico para este Ticker y este TF
                    t_bias = bias_map.get(f"{t}_{tf_main}", 0.0)
                    pf = pf_raw * (1 - (t_bias / 100))
                    cp = df_t['Close'].iloc[-1]
                    precision = 4 if cp < 20 else 2
                    pot = ((pf - cp) / cp) * 100
                    
                    # 1. Guardamos para el CSV (Formato antiguo restaurado)
                    res_row = {
                        "Fecha": datetime.now().strftime("%H:%M"),
                        "Activo": t,
                        "TF": tf_main,
                        "Precio Cierre": round(cp, precision),
                        "Predicción": round(pf, precision),
                        "Dirección": "🚀 COMPRA" if pf > cp else "📉 VENTA",
                        "Potencial %": round(pot, 2),
                        "Acuerdo %": round(av, 1),
                        "Sesgo %": round(t_bias, 2)
                    }
                    results.append(res_row)

                    # 2. ALIMENTAR EL HISTORIAL (Para que el botón de Sincronizar aprenda de estos 300)
                    nueva_fila_h = pd.DataFrame([{
                        "Fecha": res_row["Fecha"], "Activo": t, "TF": tf_main,
                        "Precio": round(cp, precision), "Predicción": round(pf, precision),
                        "Dirección": res_row["Dirección"], "Potencial %": res_row["Potencial %"],
                        "Acuerdo %": res_row["Acuerdo %"]
                    }])
                    st.session_state.historial_consultas = pd.concat([st.session_state.historial_consultas, nueva_fila_h], ignore_index=True)

            bar.progress((idx+1)/len(tickers))
        
        if results:
            df_res = pd.DataFrame(results).sort_values("Acuerdo %", ascending=False)
            st.dataframe(df_res, use_container_width=True)
            st.download_button("📥 Reporte Maestro", get_csv_download_link(df_res), f"escaneo_v6_pro_{datetime.now().strftime('%Y%m%d')}.csv")