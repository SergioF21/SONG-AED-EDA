import streamlit as st
import subprocess
import json
import pandas as pd
import os
import time
import platform
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    # Emoji simple como logo
    st.markdown("<h1 style='text-align: center;'>⚡</h1>", unsafe_allow_html=True)
with col2:
    st.title("Benchmark: SONG (GPU) vs FAISS (CPU)")
    st.markdown("**Proyecto AED** | Comparación de rendimiento en búsqueda de vecinos cercanos.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Parámetros")
k_value = st.sidebar.slider("Vecinos a buscar (K)", 1, 128, 16)
num_queries = st.sidebar.number_input("Cantidad de Consultas", 1, 10000, 100)
start_node = st.sidebar.number_input("Nodo Inicial", 0, 100000, 0)

st.sidebar.markdown("---")
st.sidebar.info("SONG: Kernel CUDA optimizado (Zhao et al.).\nFAISS: IndexFlatL2 (CPU Baseline).")

# --- RUTAS INTELIGENTES ---
# Detectamos el SO para saber si buscar .exe o no
is_windows = platform.system() == "Windows"
exe_ext = ".exe" if is_windows else ""

# Nombres base
SONG_EXE_NAME = "song" + exe_ext
FAISS_EXE_NAME = "faiss_demo" + exe_ext
GRAPH_BUILDER_EXE_NAME = "GraphBuilder" + exe_ext

# Rutas relativas (ajusta si tu estructura de carpetas es diferente)
# Asumimos que app.py está en /frontend y los ejecutables en ../backend/song/
BASE_BACKEND_DIR = os.path.join("..", "backend", "song")

PATH_SONG = os.path.join(BASE_BACKEND_DIR, SONG_EXE_NAME)
PATH_FAISS = os.path.join(BASE_BACKEND_DIR, FAISS_EXE_NAME)
PATH_GRAPH_BUILDER = os.path.join(BASE_BACKEND_DIR, GRAPH_BUILDER_EXE_NAME)

DATASET_BIN = os.path.join(BASE_BACKEND_DIR, "dataset.bin")
# FAISS suele usar el mismo dataset, o uno específico si lo generaste aparte
DATASET_FAISS = os.path.join(BASE_BACKEND_DIR, "dataset_faiss.bin") 
# Si no existe el específico de FAISS, usamos el genérico como fallback
if not os.path.exists(DATASET_FAISS) and os.path.exists(DATASET_BIN):
    DATASET_FAISS = DATASET_BIN 

GRAPH_BIN = os.path.join(BASE_BACKEND_DIR, "graph_index.bin")

# Archivos de salida (se generan donde se corre el script)
JSON_SONG = "frontend_results.json"
JSON_FAISS = "frontend_results_faiss.json"

# --- EJECUCIÓN ---
if st.sidebar.button("🚀 Ejecutar Benchmark", type="primary"):
    
    # Verificaciones previas
    missing = []
    if not os.path.exists(PATH_SONG): missing.append(f"Ejecutable SONG ({PATH_SONG})")
    if not os.path.exists(PATH_FAISS): missing.append(f"Ejecutable FAISS ({PATH_FAISS})")
    
    if missing:
        st.error("❌ Faltan archivos ejecutables:")
        for m in missing: st.write(f"- {m}")
        st.info("Por favor, compila el backend primero (nvcc para song, g++ para faiss).")
        st.stop()

    progress_bar = st.progress(0)
    status = st.empty()

    # 1. SONG (GPU)
    status.info(f"🔵 Ejecutando SONG (GPU)... K={k_value}, Q={num_queries}")
    # Argumentos: ./song dataset graph start Q K
    #  
    cmd_make_graph = [PATH_GRAPH_BUILDER, "16", str(k_value)] 
    cmd_song = [PATH_SONG, DATASET_BIN, GRAPH_BIN, str(start_node), str(num_queries), str(k_value)]
    
    try:
        # cwd="." para que el JSON se genere en la carpeta actual
        res_make_graph = subprocess.run(cmd_make_graph, capture_output=True, text=True, cwd=".")
        res_song = subprocess.run(cmd_song, capture_output=True, text=True, cwd=".")
        if res_song.returncode != 0:
            st.error("Error en SONG:")
            st.code(res_song.stderr)
            st.stop()
    except Exception as e:
        st.error(f"Excepción al lanzar SONG: {e}")
        st.stop()
    
    progress_bar.progress(50)

    # 2. FAISS (CPU)
    status.info(f"🟠 Ejecutando FAISS (CPU)... K={k_value}, Q={num_queries}")
    # Argumentos: ./faiss_demo dataset start Q K
    cmd_faiss = [PATH_FAISS, DATASET_FAISS, str(start_node), str(num_queries), str(k_value)]
    
    try:
        res_faiss = subprocess.run(cmd_faiss, capture_output=True, text=True, cwd=".")
        if res_faiss.returncode != 0:
            st.error("Error en FAISS:")
            st.code(res_faiss.stderr)
            st.stop()
    except Exception as e:
        st.error(f"Excepción al lanzar FAISS: {e}")
        st.stop()

    progress_bar.progress(100)
    status.success("¡Benchmark finalizado correctamente!")
    time.sleep(1)
    status.empty()
    progress_bar.empty()

    # --- RESULTADOS ---
    try:
        with open(JSON_SONG, 'r') as f: d_song = json.load(f)
        with open(JSON_FAISS, 'r') as f: d_faiss = json.load(f)
    except FileNotFoundError:
        st.error("No se encontraron los archivos JSON de resultados. ¿Fallaron los ejecutables silenciosamente?")
        st.stop()

    t_song = d_song.get('execution_time', 0)
    t_faiss = d_faiss.get('execution_time', 0)
    
    # Evitar división por cero
    speedup = t_faiss / t_song if t_song > 0 else 0

    # Métricas Principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Tiempo SONG (GPU)", f"{t_song:.5f} s", delta=f"-{t_faiss-t_song:.5f} s", delta_color="inverse")
    col2.metric("Tiempo FAISS (CPU)", f"{t_faiss:.5f} s")
    col3.metric("Speedup (Aceleración)", f"{speedup:.1f}x", delta="Más rápido", delta_color="normal")

    st.divider()

    # Gráfica de Barras (Plotly)
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

    # Análisis de Precisión (Recall)
    st.subheader("🔍 Análisis de Precisión (Ground Truth vs Aproximación)")
    
    # Usamos la primera query como muestra para la tabla
    if d_song['queries'] and d_faiss['queries']:
        # Selección de query para ver detalles
        query_ids = [q['query_id'] for q in d_song['queries']]
        selected_qid = st.selectbox("Seleccionar Query ID para inspeccionar:", query_ids)
        
        # Buscar datos de esa query
        res_s = next(q['results'] for q in d_song['queries'] if q['query_id'] == selected_qid)
        res_f = next(q['results'] for q in d_faiss['queries'] if q['query_id'] == selected_qid)
        
        ids_s = [r['id'] for r in res_s]
        ids_f = [r['id'] for r in res_f]
        
        # Calcular Recall (Intersección de conjuntos)
        intersection = len(set(ids_s) & set(ids_f))
        recall = (intersection / k_value) * 100
        
        c1, c2 = st.columns([1, 3])
        c1.metric("Recall (Top-K)", f"{recall:.1f}%", help="Porcentaje de vecinos correctos encontrados por SONG")
        
        # Tabla Comparativa
        # A veces SONG devuelve menos de K si no encuentra suficientes, rellenamos para la tabla
        max_len = max(len(ids_s), len(ids_f))
        ids_s += [-1] * (max_len - len(ids_s))
        ids_f += [-1] * (max_len - len(ids_f))
        dists_s = [r['distance'] for r in res_s] + [0.0] * (max_len - len(res_s))
        dists_f = [r['distance'] for r in res_f] + [0.0] * (max_len - len(res_f))

        df = pd.DataFrame({
            "Rank": range(1, max_len + 1),
            "SONG ID": ids_s,
            "SONG Dist": dists_s,
            "FAISS ID (Truth)": ids_f,
            "FAISS Dist": dists_f
        })
        
        # Resaltar coincidencias en la tabla
        def highlight(row):
            # Si coinciden los IDs (y no son relleno -1), pintamos
            color = 'background-color: #1E3A5F' if (row['SONG ID'] == row['FAISS ID (Truth)'] and row['SONG ID'] != -1) else ''
            return [color] * len(row)

        c2.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)

else:
    st.info("👈 Ajusta los parámetros en la barra lateral y presiona 'Ejecutar Benchmark' para iniciar.")

# Footer
st.markdown("---")
st.markdown("*Proyecto SONG - Estructuras de Datos Avanzadas*")
