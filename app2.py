"""
SUPERTIENDA ANALYTICS HUB + SIMULADOR PREDICTIVO WHAT-IF
Materia: Business Intelligence & Analytics - Universidad de Palermo (UP)
Docente: Ing. Matias Agustin Perez
Clase 11: Puesta en Produccion y Cloud Deploy
"""

from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -------------------------------------------------------------
# 1. CONFIGURACION DE PAGINA Y ESTILOS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Supertienda Analytics Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.0rem; font-weight: 700; color: #0A3073; margin-bottom: 2px; }
    .sub-title { font-size: 0.95rem; color: #5A5F6E; margin-bottom: 18px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0A3073; }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .badge-data {
        background-color: #EEF4FC;
        color: #0A3073;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. CARGA DE MODELO (CON CACHE DE RECURSOS)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Inicializando modelo predictivo de Machine Learning...")
def load_ml_model():
    model_path = Path(__file__).parent / "model_superstore.joblib"
    if not model_path.exists():
        model_path = Path("model_superstore.joblib")
    package = joblib.load(model_path)
    return package["pipeline"], package["metadata"]

model, metadata = load_ml_model()


# -------------------------------------------------------------
# 3. FUNCIONES DE CARGA Y ACTUALIZACION DINAMICA DE DATOS
# -------------------------------------------------------------
@st.cache_data(show_spinner="Cargando dataset base...")
def load_base_data() -> pd.DataFrame:
    data_path = Path(__file__).parent / "supertienda_curated.parquet"
    if not data_path.exists():
        data_path = Path("supertienda_curated.parquet")
    df = pd.read_parquet(data_path)
    if "fecha_pedido" in df.columns:
        df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"])
    return df

def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Lee archivos Parquet o Excel subidos por el usuario en memoria."""
    nombre = uploaded_file.name.lower()
    if nombre.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    elif nombre.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Suba archivos .parquet, .xlsx o .csv")
    
    if "fecha_pedido" in df.columns:
        df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"])
    return df


# -------------------------------------------------------------
# 4. SIDEBAR: ORIGEN DE DATOS Y FILTROS GLOBALES
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### Centro de control de datos")
    
    # Carga interactiva en vivo
    uploaded_file = st.file_uploader(
        "Actualizar datos (.parquet, .xlsx, .csv):",
        type=["parquet", "xlsx", "csv"],
        help="Sube un archivo para analizarlo en la sesion actual sin modificar el repositorio central."
    )
    
    origen_texto = ""
    if uploaded_file is not None:
        try:
            df_raw = parse_uploaded_file(uploaded_file)
            origen_texto = f"Cargado en sesion: {uploaded_file.name}"
            st.success("Dataset cargado correctamente en memoria.")
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")
            df_raw = load_base_data()
            origen_texto = "supertienda_curated.parquet (Base)"
    else:
        df_raw = load_base_data()
        origen_texto = "supertienda_curated.parquet (Base de produccion)"
        
    st.markdown(f'<div class="badge-data">Fuente: {origen_texto}</div>', unsafe_allow_html=True)
    st.caption(f"Registros activos: {len(df_raw):,} | Columnas: {df_raw.shape[1]}")

    st.divider()
    st.markdown("### Filtros analiticos")
    
    # Filtros reactivos tolerantes a la estructura
    if "anio" in df_raw.columns:
        anios_avail = sorted(df_raw["anio"].dropna().unique())
        anios_sel = st.multiselect("Anio fiscal:", anios_avail, default=anios_avail)
    else:
        anios_sel = None

    if "region" in df_raw.columns:
        reg_avail = sorted(df_raw["region"].dropna().unique())
        regiones_sel = st.multiselect("Region:", reg_avail, default=reg_avail)
    else:
        regiones_sel = None

    if "categoria" in df_raw.columns:
        cat_avail = sorted(df_raw["categoria"].dropna().unique())
        cats_sel = st.multiselect("Categoria:", cat_avail, default=cat_avail)
    else:
        cats_sel = None

    st.divider()
    st.markdown("### Modelo predictivo en produccion")
    st.write(f"**ROC-AUC Score:** `{metadata['roc_auc']}`")
    st.write(f"**Target:** {metadata['target']}")
    st.caption("Pipeline de Random Forest con ColumnTransformer en memoria RAM.")


# -------------------------------------------------------------
# 5. FILTRADO REACTIVO
# -------------------------------------------------------------
df_filtered = df_raw.copy()
if anios_sel and "anio" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["anio"].isin(anios_sel)]
if regiones_sel and "region" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["region"].isin(regiones_sel)]
if cats_sel and "categoria" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["categoria"].isin(cats_sel)]


# -------------------------------------------------------------
# 6. ENCABEZADO Y TARJETAS DE KPIS
# -------------------------------------------------------------
st.markdown('<p class="main-title">Supertienda Analytics Hub</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Plataforma de inteligencia de negocios y simulacion prescriptiva de decisiones comerciales.</p>', unsafe_allow_html=True)

if df_filtered.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados.")
    st.stop()

# Calculo de metricas
ventas_tot = df_filtered["venta"].sum() if "venta" in df_filtered.columns else 0.0
ganancia_tot = df_filtered["ganancia"].sum() if "ganancia" in df_filtered.columns else 0.0
margen_pct = (ganancia_tot / ventas_tot * 100) if ventas_tot > 0 else 0.0
pedidos_tot = df_filtered["id_pedido"].nunique() if "id_pedido" in df_filtered.columns else len(df_filtered)
ticket_prom = ventas_tot / pedidos_tot if pedidos_tot > 0 else 0.0
filas_perdida = (df_filtered["ganancia"] < 0).sum() if "ganancia" in df_filtered.columns else 0
pct_perdida = (filas_perdida / len(df_filtered) * 100) if len(df_filtered) > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventas Totales", f"${ventas_tot:,.0f}".replace(",", "."), f"{pedidos_tot:,} pedidos".replace(",", "."))
c2.metric("Ganancia Neta", f"${ganancia_tot:,.0f}".replace(",", "."), f"Margen: {margen_pct:.1f}%")
c3.metric("Ticket Promedio", f"${ticket_prom:,.0f}".replace(",", "."))
c4.metric("Ordenes con Perdida", f"{pct_perdida:.1f}%", f"{filas_perdida} pedidos", delta_color="inverse")

st.write("")


# -------------------------------------------------------------
# 7. VISTAS TEMATICAS (PESTANAS)
# -------------------------------------------------------------
tab_sim, tab_dash, tab_data, tab_model = st.tabs([
    "Simulador What-If",
    "Tablero de Rentabilidad",
    "Explorador y Descarga",
    "Ficha Tecnica del Modelo"
])

# ----------------- PESTANA 1: SIMULADOR WHAT-IF -----------------
with tab_sim:
    st.subheader("Simulador de Cotizaciones y Politicas de Descuento")
    st.markdown("Evalua en tiempo real si una cotizacion comercial generara margen positivo o destruccion neta de capital antes de emitir la orden.")

    col_sim_in, col_sim_out = st.columns([6, 6])

    with col_sim_in:
        st.markdown("##### 1. Parametros de la operacion")
        c_cat, c_sub = st.columns(2)
        with c_cat:
            cat_choice = st.selectbox("Categoria:", metadata["categories"]["categoria"])
        with c_sub:
            if "categoria" in df_raw.columns and "subcategoria" in df_raw.columns:
                subcats_avail = df_raw[df_raw["categoria"] == cat_choice]["subcategoria"].unique().tolist()
            else:
                subcats_avail = metadata["categories"]["subcategoria"]
            subcat_choice = st.selectbox("Subcategoria:", sorted(subcats_avail))

        c_reg, c_seg = st.columns(2)
        with c_reg:
            reg_choice = st.selectbox("Region:", metadata["categories"]["region"])
        with c_seg:
            seg_choice = st.selectbox("Segmento:", metadata["categories"]["segmento"])

        c_env1, c_env2 = st.columns(2)
        with c_env1:
            env_choice = st.selectbox("Forma de envio:", metadata["categories"]["forma_envio"])
        with c_env2:
            dias_env_choice = st.slider("Dias de envio estimados:", 0, 10, 4)

        st.markdown("##### 2. Variables comerciales")
        c_vta, c_cant = st.columns(2)
        with c_vta:
            venta_choice = st.number_input("Facturacion estimada ($):", min_value=10.0, max_value=50000.0, value=3500.0, step=100.0)
        with c_cant:
            cant_choice = st.number_input("Cantidad de unidades:", min_value=1, max_value=100, value=5, step=1)

        desc_choice = st.slider(
            "Tasa de descuento ofrecida (%):",
            min_value=0, max_value=80, value=15, step=1,
            help="Prueba superar el 20% para ver la sensibilidad de riesgo proyectada."
        ) / 100.0

    # Inferencia
    input_df = pd.DataFrame([{
        "categoria": cat_choice,
        "subcategoria": subcat_choice,
        "region": reg_choice,
        "segmento": seg_choice,
        "forma_envio": env_choice,
        "venta": float(venta_choice),
        "cantidad": int(cant_choice),
        "descuento": float(desc_choice),
        "dias_envio": int(dias_env_choice)
    }])

    prob_perdida = float(model.predict_proba(input_df)[0][1])

    with col_sim_out:
        st.markdown("##### 3. Dictamen del modelo de riesgo")
        st.write("")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_perdida * 100,
            title={'text': "Probabilidad de Perdida Neta (%)"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#0A3073"},
                'steps': [
                    {'range': [0, 30], 'color': "#D1FAE5"},
                    {'range': [30, 60], 'color': "#FEF3C7"},
                    {'range': [60, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prob_perdida < 0.30:
            st.success("**Operacion recomendada:** Riesgo bajo. La transaccion preserva un margen saludable.")
        elif prob_perdida < 0.60:
            st.warning("**Requiere aprobacion gerencial:** Riesgo moderado. El descuento roza el limite de rentabilidad.")
        else:
            st.error("**Operacion no recomendada:** Riesgo critico de perdida neta. Ajustar descuento hacia abajo.")


# ----------------- PESTANA 2: TABLERO DESCRIPTIVO -----------------
with tab_dash:
    st.subheader("Auditoria Historica de Rentabilidad y Descuentos")
    if "descuento" in df_filtered.columns and "ganancia" in df_filtered.columns:
        fig_scat = px.scatter(
            df_filtered,
            x="descuento",
            y="ganancia",
            color="categoria" if "categoria" in df_filtered.columns else None,
            size="venta" if "venta" in df_filtered.columns else None,
            hover_data=["subcategoria", "region"] if "subcategoria" in df_filtered.columns else None,
            title="Ganancia Neta vs. Descuento Aplicado en Transacciones Activas"
        )
        fig_scat.add_hline(y=0, line_dash="dash", line_color="red")
        fig_scat.update_layout(template="plotly_white")
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info("El dataset activo no contiene las columnas requeridas ('descuento', 'ganancia').")


# ----------------- PESTANA 3: EXPLORADOR Y DESCARGA -----------------
with tab_data:
    st.subheader("Explorador de Transacciones Filtradas")
    cols_show = [c for c in ["id_pedido", "fecha_pedido", "nombre_cliente", "region", "categoria", "subcategoria", "venta", "ganancia", "descuento"] if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_show], use_container_width=True, hide_index=True)

    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar datos filtrados en CSV",
        data=csv_data,
        file_name="supertienda_reporte_filtrado.csv",
        mime="text/csv"
    )


# ----------------- PESTANA 4: FICHA TECNICA -----------------
with tab_model:
    st.subheader("Ficha Tecnica de Arquitectura del Modelo")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("**Algoritmo:** Random Forest Classifier (Pipeline de Scikit-Learn)")
        st.write(f"**ROC-AUC Score:** `{metadata['roc_auc']}`")
        st.write("**Features categoricas:**", metadata.get("cat_features", []))
        st.write("**Features numericas:**", metadata.get("num_features", []))
    with col_m2:
        st.info("""
        **Politica de Despliegue en la Nube:**
        - **Inferencia en memoria RAM:** El pipeline serializado con joblib se carga una sola vez al levantar el contenedor mediante `@st.cache_resource`.
        - **Persistencia en uploader:** Al cargar archivos interactivos, los datos residen en el buffer de sesion sin modificar el archivo base del repositorio.
        """)

st.divider()
st.caption("Universidad de Palermo • Business Intelligence & Analytics • Clase 11: Cloud Deploy • Ing. Matias A. Perez")
