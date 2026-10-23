# Supertienda analytics hub y simulador predictivo

Aplicacion analitica interactiva y simulador predictivo what-if desarrollada con Streamlit, Plotly y Scikit-Learn para la toma de decisiones comerciales.

## Descripcion del proyecto

Esta solucion combina analitica descriptiva, diagnostica y prescriptiva para una cadena minorista multiregion:
- **Monitoreo ejecutivo:** Evaluacion en tiempo real de ventas totales, ganancia operativa, margen comercial y tasa de devoluciones.
- **Simulador what-if:** Modelo de clasificacion y regresion entrenado con Scikit-Learn que evalua el impacto proyectado en margen y rentabilidad al modificar parametros comerciales (descuentos, modos de envio, categorias).
- **Actualizacion dinamica de datos:** Centro de carga en barra lateral con soporte para archivos `.parquet`, `.xlsx` y `.csv` para analisis interactivo en memoria de sesion.
- **Explorador y descarga:** Filtros multidimensionales y exportacion de datos filtrados y escenarios simulados.

## Estructura del repositorio

```text
app_supertienda_produccion/
├── app.py                      # Punto de entrada de la aplicacion Streamlit
├── requirements.txt            # Dependencias de Python para el contenedor
├── supertienda_curated.parquet # Dataset base en formato columnar optimizado
├── model_superstore.joblib     # Modelo predictivo entrenado serializado
├── .streamlit/
│   └── config.toml             # Configuracion de estilos institucionales y servidor
├── .gitignore                  # Exclusion de entornos y archivos temporales
└── README.md                   # Documentacion del proyecto
```

## Ejecucion en entorno local

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/supertienda-analytics-hub.git
cd supertienda-analytics-hub
```

### 2. Crear y activar entorno virtual
En Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

En Linux o macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar librerias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Iniciar la aplicacion
```bash
streamlit run app.py
```
La aplicacion estara disponible en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Conectar tu cuenta de GitHub con [Streamlit Community Cloud](https://share.streamlit.io).
2. Seleccionar este repositorio y la rama `main`.
3. Especificar `app.py` como el archivo principal.
4. Presionar **Deploy**. La aplicacion quedara publicada bajo una URL publica HTTPS.
