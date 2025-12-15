import streamlit as st
import subprocess
import json
import pandas as pd
import os
import time
import platform
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="SONG vs FAISS", page_icon="⚡", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
    }
    .stDataFrame {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("<h1 style='text-align: center;'>⚡</h1>", unsafe_allow_html=True)
with col2:
    st.title("Benchmark: SONG (GPU) vs FAISS (CPU)")
    st.markdown("**Proyecto AED** | Comparación de rendimiento en búsqueda de vecinos cercanos.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Parámetros")
k_value = st.sidebar.slider("Vecinos a buscar (K)", 1, 128, 16)
num_queries = st.sidebar.number_input("Cantidad de Consultas", 1, 1000000000000, 100)
start_node = st.sidebar.number_input("Nodo Inicial", 0, 100000, 0)

st.sidebar.markdown("---")
st.sidebar.info("SONG: Kernel CUDA optimizado.\nFAISS: IndexFlatL2 (CPU Baseline).")

# --- RUTAS ---
is_windows = platform.system() == "Windows"
exe_ext = ".exe" if is_windows else ""

SONG_EXE_NAME = "song" + exe_ext
FAISS_EXE_NAME = "faiss_demo" + exe_ext
GRAPH_BUILDER_EXE_NAME = "GraphBuilder" + exe_ext

BASE_BACKEND_DIR = os.path.join("..", "backend", "song")

PATH_SONG = os.path.join(BASE_BACKEND_DIR, SONG_EXE_NAME)
PATH_FAISS = os.path.join(BASE_BACKEND_DIR, FAISS_EXE_NAME)
PATH_GRAPH_BUILDER = os.path.join(BASE_BACKEND_DIR, GRAPH_BUILDER_EXE_NAME)

DATASET_BIN = os.path.join(BASE_BACKEND_DIR, "dataset.bin")
DATASET_FAISS = os.path.join(BASE_BACKEND_DIR, "dataset_faiss.bin") 
if not os.path.exists(DATASET_FAISS) and os.path.exists(DATASET_BIN):
    DATASET_FAISS = DATASET_BIN 

GRAPH_BIN = os.path.join(BASE_BACKEND_DIR, "graph_index.bin")

JSON_SONG = "frontend_results.json"
JSON_FAISS = "frontend_results_faiss.json"

# --- LÓGICA DE ESTADO (SESSION STATE) ---
# Inicializamos variables en memoria para que no se borren al interactuar
if 'benchmark_data' not in st.session_state:
    st.session_state.benchmark_data = None

# --- EJECUCIÓN DEL BENCHMARK ---
if st.sidebar.button("🚀 Ejecutar Benchmark", type="primary"):
    
    # 1. Verificaciones
    missing = []
    if not os.path.exists(PATH_SONG): missing.append(f"SONG ({PATH_SONG})")
    if not os.path.exists(PATH_FAISS): missing.append(f"FAISS ({PATH_FAISS})")
    
    if missing:
        st.error("Faltan archivos ejecutables:")
        for m in missing: st.write(f"- {m}")
        st.stop()

    progress_bar = st.progress(0)
    status = st.empty()

    # 2. SONG (GPU)
    status.info(f"🔵 Ejecutando SONG (GPU)... K={k_value}")
    cmd_song = [PATH_SONG, DATASET_BIN, GRAPH_BIN, str(start_node), str(num_queries), str(k_value)]
    
    try:
        subprocess.run([PATH_GRAPH_BUILDER, "16", str(k_value)], cwd=".")        
        res_song = subprocess.run(cmd_song, capture_output=True, text=True, cwd=".")
        if res_song.returncode != 0:
            st.error(f"Error SONG: {res_song.stderr}")
            st.stop()
    except Exception as e:
        st.error(f"Error lanzando SONG: {e}")
        st.stop()
    
    progress_bar.progress(50)

    # 3. FAISS (CPU)
    status.info(f"🟠 Ejecutando FAISS (CPU)...")
    cmd_faiss = [PATH_FAISS, DATASET_FAISS, str(start_node), str(num_queries), str(k_value)]
    
    try:
        res_faiss = subprocess.run(cmd_faiss, capture_output=True, text=True, cwd=".")
        if res_faiss.returncode != 0:
            st.error(f"Error FAISS: {res_faiss.stderr}")
            st.stop()
    except Exception as e:
        st.error(f"Error lanzando FAISS: {e}")
        st.stop()

    progress_bar.progress(100)
    status.success("¡Benchmark finalizado!")
    time.sleep(0.5)
    status.empty()
    progress_bar.empty()

    # 4. CARGA DE DATOS A SESSION STATE
    try:
        with open(JSON_SONG, 'r') as f: d_song = json.load(f)
        with open(JSON_FAISS, 'r') as f: d_faiss = json.load(f)
        
        # Guardamos en memoria persistente
        st.session_state.benchmark_data = {
            'song': d_song,
            'faiss': d_faiss,
            'k': k_value
        }
        # Forzamos recarga para asegurar que se pinte la UI de abajo
        st.rerun()

    except Exception as e:
        st.error(f"Error leyendo JSONs: {e}")
        st.stop()

# --- RENDERIZADO DE RESULTADOS ---
# Esta parte se ejecuta si ya existen datos en memoria, sin importar si presionaste el botón o no
if st.session_state.benchmark_data:
    data = st.session_state.benchmark_data
    d_song = data['song']
    d_faiss = data['faiss']
    current_k = data['k']

    # Métricas de Tiempo
    t_song = d_song.get('execution_time', 0)
    t_faiss = d_faiss.get('execution_time', 0)
    speedup = t_faiss / t_song if t_song > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Tiempo SONG (GPU)", f"{t_song:.5f} s", delta=f"-{t_faiss-t_song:.5f} s", delta_color="inverse")
    col2.metric("Tiempo FAISS (CPU)", f"{t_faiss:.5f} s")
    col3.metric("Speedup", f"{speedup:.1f}x", delta="Más rápido", delta_color="normal")

    # Gráfica de Barras (Restaurada)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['SONG (GPU)', 'FAISS (CPU)'],
        x=[t_song, t_faiss],
        orientation='h',
        marker_color=['#00CC96', '#EF553B'],
        text=[f"{t_song:.4f}s", f"{t_faiss:.4f}s"],
        textposition='auto',
    ))
    fig.update_layout(
        title="Tiempo de Ejecución (Menos es mejor)",
        xaxis_title="Segundos",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Análisis Global
    st.subheader("🔍 Análisis de Precisión Global")

    if 'queries' in d_song and 'queries' in d_faiss:
        
        all_recalls = []
        summary_data = []
        
        # Preprocesamiento de datos
        # Creamos un diccionario de FAISS para búsqueda rápida por ID
        faiss_dict = {q['query_id']: q for q in d_faiss['queries']}
        
        for q_s in d_song['queries']:
            q_id = q_s['query_id']
            q_f = faiss_dict.get(q_id)
            
            if q_f:
                ids_s = [r['id'] for r in q_s['results']]
                ids_f = [r['id'] for r in q_f['results']]
                
                intersection = len(set(ids_s) & set(ids_f))
                recall_pct = (intersection / current_k) * 100
                all_recalls.append(recall_pct)
                
                summary_data.append({
                    "Query ID": q_id,
                    "Recall (%)": recall_pct,
                    "Top-1 Match": (ids_s[0] == ids_f[0]) if ids_s and ids_f else False
                })

        # Gráficos Resumen
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
        
        m_col1, m_col2 = st.columns([1, 3])
        m_col1.metric("Recall Promedio", f"{avg_recall:.2f}%")
        
        fig_hist = px.histogram(all_recalls, nbins=20, labels={'value': 'Recall (%)'}, 
                                title="Distribución de Precisión",
                                color_discrete_sequence=['#00CC96'])
        fig_hist.update_layout(showlegend=False, height=250, margin=dict(l=20, r=20, t=30, b=20))
        m_col2.plotly_chart(fig_hist, use_container_width=True)

        # Tabla Resumen Scrollable
        st.write("### 📋 Resumen por Consulta")
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(
            df_summary.style.background_gradient(cmap='RdYlGn', subset=['Recall (%)'], vmin=0, vmax=100), 
            use_container_width=True, 
            height=250
        )

        # --- SECCIÓN INTERACTIVA (CAUSA DEL PROBLEMA ANTERIOR) ---
        st.write("### 🔬 Detalles Profundos (Vecino a Vecino)")
        st.info("Selecciona un ID para comparar:")

        # Lista de IDs disponibles
        query_ids = [q['Query ID'] for q in summary_data]
        
        # El selectbox ahora funciona porque 'st.session_state.benchmark_data' persiste tras el rerun
        selected_qid = st.selectbox("ID de Query:", query_ids)
        
        if selected_qid is not None:
            # Recuperar datos específicos de la memoria
            q_s_raw = next(q for q in d_song['queries'] if q['query_id'] == selected_qid)
            q_f_raw = faiss_dict.get(selected_qid)
            
            res_s = q_s_raw['results']
            res_f = q_f_raw['results']
            
            max_len = max(len(res_s), len(res_f))
            
            # Construcción segura de listas
            ids_s_list = ([r['id'] for r in res_s] + [-1] * max_len)[:max_len]
            ids_f_list = ([r['id'] for r in res_f] + [-1] * max_len)[:max_len]
            dists_s = ([r['distance'] for r in res_s] + [0.0] * max_len)[:max_len]
            dists_f = ([r['distance'] for r in res_f] + [0.0] * max_len)[:max_len]

            df_detail = pd.DataFrame({
                "Rank": range(1, max_len + 1),
                "SONG ID": ids_s_list,
                "SONG Dist": dists_s,
                "FAISS ID (Truth)": ids_f_list,
                "FAISS Dist": dists_f
            })

            def highlight_matches(row):
                # Pintar azul si coinciden, rojo suave si difieren
                if row['SONG ID'] == row['FAISS ID (Truth)'] and row['SONG ID'] != -1:
                    return ['background-color: #1E3A5F'] * len(row)
                else:
                    return [''] * len(row)

            st.dataframe(df_detail.style.apply(highlight_matches, axis=1), use_container_width=True)

    else:
        st.warning("JSON sin detalle de queries.")

else:
    # Pantalla inicial (si no hay datos en memoria)
    st.info("👈 Ajusta los parámetros y ejecuta el benchmark para ver resultados.")