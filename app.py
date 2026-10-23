"""
PASO 6: APLICACIÓN COMPLETA OPTIMIZADA CON CACHÉ (Estándar Profesional)
Concepto nuevo: @st.cache_data y st.set_page_config().
Cero HTML/CSS. 100% Código Streamlit Nativo.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. Configuración de página
st.set_page_config(page_title="Supertienda Analytics", layout="wide")

# 2. Carga con Caché para velocidad en producción
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    path = Path(__file__).parent / "supertienda_curated.parquet"
    return pd.read_parquet(path)

df = cargar_datos()

# 3. Sidebar con Filtros
st.sidebar.title("Filtros de Control")

regiones_opt = sorted(df["region"].unique())
regiones_sel = st.sidebar.multiselect("Región Comercial:", regiones_opt, default=regiones_opt)

cats_opt = sorted(df["categoria"].unique())
cats_sel = st.sidebar.multiselect("Categoría de Producto:", cats_opt, default=cats_opt)

# 4. Filtrado en Memoria
df_filtrado = df[
    (df["region"].isin(regiones_sel)) & 
    (df["categoria"].isin(cats_sel))
]

# 5. Header Principal
st.title("Supertienda - Dashboard Analítico Ejecutivo")
st.caption("Monitoreo dinámico en tiempo real de ventas, rentabilidad neta y pedidos.")

if df_filtrado.empty:
    st.warning("No hay transacciones para la combinación de filtros seleccionada.")
    st.stop()

# 6. Pestañas de Navegación
tab1, tab2, tab3 = st.tabs(["Métricas & Tendencias", "Diagnóstico de Margen", "Explorador de Datos"])

# --- TAB 1: VISIÓN GENERAL ---
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    ventas = df_filtrado["venta"].sum()
    ganancia = df_filtrado["ganancia"].sum()
    pedidos = df_filtrado["id_pedido"].nunique()
    margen_pct = (ganancia / ventas * 100) if ventas > 0 else 0

    c1.metric("Ventas Totales", f"${ventas:,.0f}")
    c2.metric("Ganancia Neta", f"${ganancia:,.0f}", delta=f"{margen_pct:.1f}% margen")
    c3.metric("Total Pedidos", f"{pedidos:,}")
    c4.metric("Clientes Únicos", f"{df_filtrado['id_cliente'].nunique():,}")

    st.divider()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Ventas por Categoría")
        df_cat = df_filtrado.groupby("categoria", observed=True)["venta"].sum().reset_index()
        fig1 = px.bar(df_cat, x="categoria", y="venta", color="categoria")
        st.plotly_chart(fig1, use_container_width=True)

    with col_t2:
        st.subheader("Participación por Segmento")
        df_seg = df_filtrado.groupby("segmento", observed=True)["venta"].sum().reset_index()
        fig2 = px.pie(df_seg, names="segmento", values="venta", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB 2: RENTABILIDAD & DESCUENTOS ---
with tab2:
    st.subheader("Análisis de Descuentos vs Ganancia")
    fig3 = px.scatter(
        df_filtrado, 
        x="descuento", 
        y="ganancia", 
        color="categoria", 
        size="venta",
        hover_data=["subcategoria", "nombre_cliente"]
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: TABLA DE DATOS ---
with tab3:
    st.subheader("Registros Filtrados")
    cols_mostrar = ["id_pedido", "fecha_pedido", "nombre_cliente", "region", "categoria", "subcategoria", "venta", "ganancia"]
    st.dataframe(df_filtrado[cols_mostrar], use_container_width=True, hide_index=True)

    csv = df_filtrado[cols_mostrar].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar Dataset Filtrado en CSV",
        data=csv,
        file_name="supertienda_filtrado.csv",
        mime="text/csv"
    )
