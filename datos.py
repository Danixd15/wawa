# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Inventory Intelligence Framework",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Framework de Optimización de Inventarios")
st.caption(
    "Pronóstico mensual + selección automática del mejor método por producto + simulación + optimización de inventarios"
)

METODOS_PRONOSTICO = [
    "Naive",
    "Promedio móvil",
    "SES",
    "Regresión lineal",
    "ARIMA",
    "SARIMA",
    "Holt-Winters",
    "Croston",
]


# =========================================================
# FUNCIONES DE DATOS
# =========================================================
def convertir_a_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte cualquier base diaria/semanal/mensual a demanda mensual por producto."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str)
    df["demand_real"] = pd.to_numeric(df["demand_real"], errors="coerce").fillna(0)
    df["demand_real"] = df["demand_real"].clip(lower=0)
    df = df.dropna(subset=["date"])

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df_mensual = (
        df.groupby(["product_id", "date"], as_index=False)["demand_real"]
        .sum()
        .sort_values(["product_id", "date"])
        .reset_index(drop=True)
    )

    if df_mensual.empty:
        raise ValueError("No hay datos válidos después de convertir la información a meses.")

    return df_mensual


def generar_demanda_sintetica(n_productos: int = 5, meses: int = 36, seed: int = 42) -> pd.DataFrame:
    """Genera demanda mensual sintética para pruebas."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(start="2023-01-01", periods=meses, freq="MS")
    dataframes = []

    for i in range(1, n_productos + 1):
        producto = f"PROD_{i:03d}"
        base = rng.integers(500, 2500)
        tendencia = rng.uniform(-10, 30)
        estacionalidad = rng.uniform(100, 400)
        ruido = rng.normal(0, base * 0.15, meses)
        tiempo = np.arange(meses)

        demanda = base + tendencia * tiempo + estacionalidad * np.sin(2 * np.pi * tiempo / 12) + ruido
        demanda = np.maximum(0, np.round(demanda)).astype(int)

        if i % 4 == 0:
            mascara_intermitente = rng.random(meses) < 0.45
            demanda = np.where(mascara_intermitente, 0, demanda)

        dataframes.append(
            pd.DataFrame(
                {
                    "date": fechas,
                    "product_id": producto,
                    "demand_real": demanda,
                }
            )
        )

    return pd.concat(dataframes, ignore_index=True)


def leer_archivo_subido(uploaded_file) -> pd.DataFrame:
    """Lee CSV o Excel, normaliza columnas y agrupa la demanda por mes."""
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Sube un archivo CSV o Excel.")

    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "período": "date",
        "día": "date",
        "dia": "date",
        "producto": "product_id",
        "sku": "product_id",
        "id_producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "demanda": "demand_real",
        "venta": "demand_real",
        "ventas": "demand_real",
        "cantidad": "demand_real",
        "unidades": "demand_real",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    columnas_requeridas = ["date", "product_id", "demand_real"]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(faltantes)
            + ". Usa columnas: date, product_id, demand_real. "
            + "También puede reconocer nombres como fecha, mes, producto, sku, ventas o demanda."
        )

    df = df[columnas_requeridas].copy()
    return convertir_a_mensual(df)
