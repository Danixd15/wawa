import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")







# =========================================================
# SIMULACIÓN DE INVENTARIO MENSUAL
# =========================================================
@dataclass
class ParametrosInventario:
    initial_stock: int
    lead_time_months: int
    review_period_months: int
    ss_months: int
    q_fixed: int
    lot_size: int
    cost_order: float
    cost_holding_month: float
    cost_stockout: float


def redondear_lote(cantidad: float, lote: int) -> int:
    if cantidad <= 0:
        return 0
    lote = max(1, int(lote))
    return int(math.ceil(cantidad / lote) * lote)


def simular_producto(df_producto: pd.DataFrame, politica: str, p: ParametrosInventario) -> pd.DataFrame:
    df_producto = df_producto.sort_values("date").reset_index(drop=True).copy()
    stock_fisico = float(p.initial_stock)
    pipeline = {}
    resultados = []

    demanda_promedio_mensual = max(0.01, df_producto["demand_forecast"].mean())

    for t, fila in df_producto.iterrows():
        llegada = pipeline.pop(t, 0)
        stock_fisico += llegada

        demanda_durante_lead_time = demanda_promedio_mensual * p.lead_time_months
        stock_seguridad = demanda_promedio_mensual * p.ss_months
        punto_reorden = demanda_durante_lead_time + stock_seguridad
        nivel_objetivo = demanda_promedio_mensual * (
            p.lead_time_months + p.review_period_months + p.ss_months
        )

        posicion_inventario = stock_fisico + sum(pipeline.values())
        orden = 0

        if politica == "RS - revisión periódica":
            if t % p.review_period_months == 0:
                orden = max(0, nivel_objetivo - posicion_inventario)
        elif politica == "sS - punto de reorden y nivel máximo":
            if posicion_inventario <= punto_reorden:
                orden = max(0, nivel_objetivo - posicion_inventario)
        elif politica == "sQ - punto de reorden y cantidad fija":
            if posicion_inventario <= punto_reorden:
                orden = p.q_fixed

        orden = redondear_lote(orden, p.lot_size)

        if orden > 0:
            mes_llegada = t + p.lead_time_months
            pipeline[mes_llegada] = pipeline.get(mes_llegada, 0) + orden

        demanda_real = float(fila["demand_real"])
        venta_real = min(stock_fisico, demanda_real)
        venta_perdida = max(0, demanda_real - stock_fisico)
        stock_fisico -= venta_real

        resultados.append(
            {
                "date": fila["date"],
                "product_id": fila["product_id"],
                "method_used": fila.get("method_used", ""),
                "demand_real": demanda_real,
                "demand_forecast": fila["demand_forecast"],
                "inventory_level": stock_fisico,
                "inventory_position": posicion_inventario,
                "order_placed": orden,
                "arrivals": llegada,
                "sales_real": venta_real,
                "sales_lost": venta_perdida,
                "reorder_point_s": punto_reorden,
                "target_level_S": nivel_objetivo,
                "is_stockout": int(venta_perdida > 0),
            }
        )

    return pd.DataFrame(resultados)


def calcular_kpis(df_sim: pd.DataFrame, p: ParametrosInventario) -> dict:
    demanda_total = df_sim["demand_real"].sum()
    ventas_perdidas = df_sim["sales_lost"].sum()
    ordenes = (df_sim["order_placed"] > 0).sum()
    inventario_promedio = df_sim["inventory_level"].mean()

    fill_rate = 1 - ventas_perdidas / demanda_total if demanda_total > 0 else 1
    costo_ordenar = ordenes * p.cost_order
    costo_mantener = df_sim["inventory_level"].sum() * p.cost_holding_month
    costo_quiebre = ventas_perdidas * p.cost_stockout
    costo_total = costo_ordenar + costo_mantener + costo_quiebre

    return {
        "fill_rate": fill_rate,
        "avg_inventory": inventario_promedio,
        "lost_sales_units": ventas_perdidas,
        "stockout_months": int(df_sim["is_stockout"].sum()),
        "orders": int(ordenes),
        "ordering_cost": costo_ordenar,
        "holding_cost": costo_mantener,
        "stockout_cost": costo_quiebre,
        "total_cost": costo_total,
    }


def optimizar_stock_seguridad(
    df_producto: pd.DataFrame,
    politica: str,
    p_base: ParametrosInventario,
    ss_max: int,
) -> pd.DataFrame:
    filas = []

    for ss in range(0, ss_max + 1):
        p = ParametrosInventario(
            initial_stock=p_base.initial_stock,
            lead_time_months=p_base.lead_time_months,
            review_period_months=p_base.review_period_months,
            ss_months=ss,
            q_fixed=p_base.q_fixed,
            lot_size=p_base.lot_size,
            cost_order=p_base.cost_order,
            cost_holding_month=p_base.cost_holding_month,
            cost_stockout=p_base.cost_stockout,
        )
        sim = simular_producto(df_producto, politica, p)
        kpis = calcular_kpis(sim, p)
        filas.append({"ss_months": ss, **kpis})

    return pd.DataFrame(filas)


# =========================================================
# VISUALIZACIONES
# =========================================================
def grafico_forecast(df_producto: pd.DataFrame) -> go.Figure:
    metodo = df_producto["method_used"].iloc[0] if "method_used" in df_producto.columns else ""

    df_hist = df_producto[df_producto.get("tipo_periodo", "Histórico") == "Histórico"].copy()
    df_future = df_producto[df_producto.get("tipo_periodo", "Histórico") == "Pronóstico futuro"].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist["demand_real"],
            mode="lines+markers",
            name="Demanda real mensual histórica",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist["demand_forecast"],
            mode="lines+markers",
            name=f"Ajuste del pronóstico ({metodo})",
        )
    )

    if not df_future.empty:
        fig.add_trace(
            go.Scatter(
                x=df_future["date"],
                y=df_future["demand_forecast"],
                mode="lines+markers",
                name=f"Pronóstico futuro ({metodo})",
                line={"dash": "dash"},
            )
        )

    fig.update_layout(
        title=f"Demanda mensual histórica y pronóstico futuro - Método usado: {metodo}",
        xaxis_title="Mes",
        yaxis_title="Unidades",
        hovermode="x unified",
    )
    return fig


def grafico_inventario(df_sim: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_sim["date"], y=df_sim["inventory_level"], name="Inventario", mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sim["date"],
            y=df_sim["reorder_point_s"],
            name="Punto s",
            mode="lines",
            line={"dash": "dot"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=df_sim["date"], y=df_sim["demand_real"], name="Demanda mensual", opacity=0.35),
        secondary_y=True,
    )

    pedidos = df_sim[df_sim["order_placed"] > 0]
    fig.add_trace(
        go.Scatter(
            x=pedidos["date"],
            y=pedidos["order_placed"],
            name="Pedido generado",
            mode="markers",
            marker={"size": 10, "symbol": "triangle-up"},
        ),
        secondary_y=True,
    )

    fig.update_layout(title="Simulación mensual de inventario", hovermode="x unified")
    fig.update_yaxes(title_text="Inventario", secondary_y=False)
    fig.update_yaxes(title_text="Demanda / Pedidos", secondary_y=True)
    return fig


def grafico_tradeoff(df_opt: pd.DataFrame) -> go.Figure:
    mejor = df_opt.loc[df_opt["total_cost"].idxmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["total_cost"], mode="lines+markers", name="Costo total"))
    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["holding_cost"], mode="lines", name="Costo mantener"))
    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["stockout_cost"], mode="lines", name="Costo quiebre"))
    fig.add_vline(
        x=int(mejor["ss_months"]),
        line_dash="dash",
        annotation_text=f"Óptimo: {int(mejor['ss_months'])} meses",
    )
    fig.update_layout(
        title="Trade-off de costos",
        xaxis_title="Meses de stock de seguridad",
        yaxis_title="Costo",
        hovermode="x unified",
    )
    return fig


def formatear_comparacion(df_comparacion: pd.DataFrame) -> pd.DataFrame:
    df = df_comparacion.copy()
    df["wMAPE"] = df["wMAPE"].map(lambda x: f"{x:.2%}")
    df["Bias"] = df["Bias"].map(lambda x: f"{x:.2%}")
    df["MAE"] = df["MAE"].map(lambda x: f"{x:,.2f}")
    df["Resultado"] = np.where(df["Es mejor"], "✅ Mejor", "")
    return df[["Producto", "Método", "wMAPE", "Bias", "MAE", "Resultado"]]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("1. Carga de datos")
modo_datos = st.sidebar.radio("Modo de datos", ["Generar datos sintéticos", "Subir CSV/Excel"])

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    meses = st.sidebar.slider("Meses de historial", 12, 84, 36)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)
    df_real = generar_demanda_sintetica(n_productos=n_productos, meses=meses, seed=seed)
else:
    archivo = st.sidebar.file_uploader("Sube tu archivo", type=["csv", "xlsx", "xls"])
    if archivo is None:
        st.info(
            "Sube un CSV o Excel con columnas: date, product_id, demand_real. "
            "Si tus datos son diarios, la app los agrupará por mes."
        )
        st.stop()

    try:
        df_real = leer_archivo_subido(archivo)
    except Exception as e:
        st.error(str(e))
        st.stop()

st.sidebar.header("2. Pronóstico mensual")
modo_pronostico = st.sidebar.selectbox(
    "Selección del método",
    ["Automático: mejor método por producto", "Manual: elegir un método"],
)

ultima_fecha_historica = pd.to_datetime(df_real["date"].max()).to_period("M").to_timestamp()
fecha_fin_pronostico = st.sidebar.date_input(
    "Pronosticar hasta",
    value=pd.Timestamp("2026-12-01"),
    min_value=ultima_fecha_historica.date(),
)
fecha_fin_pronostico = pd.to_datetime(fecha_fin_pronostico).to_period("M").to_timestamp()

df_forecast_auto, df_comparacion = generar_forecast_mejor_por_producto(
    df_real, fecha_fin_pronostico=fecha_fin_pronostico
)

if modo_pronostico == "Manual: elegir un método":
    metodo_manual = st.sidebar.selectbox("Método manual", METODOS_PRONOSTICO)
    df_forecast = generar_forecast(df_real, metodo_manual, fecha_fin_pronostico=fecha_fin_pronostico)
else:
    metodo_manual = None
    df_forecast = df_forecast_auto

productos = sorted(df_forecast["product_id"].unique())
producto_sel = st.sidebar.selectbox("Producto a visualizar", productos)

sub_comparacion_producto = df_comparacion[df_comparacion["Producto"] == producto_sel].copy()
mejor_metodo_producto = sub_comparacion_producto.loc[sub_comparacion_producto["Es mejor"], "Método"].iloc[0]
mejor_wmape_producto = sub_comparacion_producto.loc[sub_comparacion_producto["Es mejor"], "wMAPE"].iloc[0]

if modo_pronostico == "Automático: mejor método por producto":
    st.sidebar.success(f"Método elegido para {producto_sel}: {mejor_metodo_producto}")
else:
    st.sidebar.info(f"Mejor método para {producto_sel}: {mejor_metodo_producto}")

st.sidebar.header("3. Política de inventario mensual")
politica = st.sidebar.selectbox(
    "Política",
    [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
        "sQ - punto de reorden y cantidad fija",
    ],
)

initial_stock = st.sidebar.number_input("Stock inicial", min_value=0, value=1000, step=100)
lead_time_months = st.sidebar.number_input("Lead time / tiempo de entrega (meses)", min_value=1, value=1, step=1)
review_period_months = st.sidebar.number_input("Periodo de revisión R (meses)", min_value=1, value=1, step=1)
ss_months = st.sidebar.number_input("Stock de seguridad inicial (meses)", min_value=0, value=1, step=1)
q_fixed = st.sidebar.number_input("Cantidad fija Q", min_value=1, value=1000, step=100)
lot_size = st.sidebar.number_input("Tamaño de lote / empaque", min_value=1, value=1, step=1)

st.sidebar.header("4. Costos")
cost_order = st.sidebar.number_input("Costo por orden", min_value=0.0, value=200.0, step=10.0)
cost_holding_month = st.sidebar.number_input("Costo mensual de mantener 1 unidad", min_value=0.0, value=1.5, step=0.5)
cost_stockout = st.sidebar.number_input("Costo por unidad perdida", min_value=0.0, value=500.0, step=10.0)
ss_max = st.sidebar.slider("Máximo SS para optimizar (meses)", 1, 24, 6)

parametros = ParametrosInventario(
    initial_stock=int(initial_stock),
    lead_time_months=int(lead_time_months),
    review_period_months=int(review_period_months),
    ss_months=int(ss_months),
    q_fixed=int(q_fixed),
    lot_size=int(lot_size),
    cost_order=float(cost_order),
    cost_holding_month=float(cost_holding_month),
    cost_stockout=float(cost_stockout),
)


# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================
sub_forecast = df_forecast[df_forecast["product_id"] == producto_sel].copy()
metodo_usado = sub_forecast["method_used"].iloc[0]
sub_sim = simular_producto(sub_forecast, politica, parametros)
kpis = calcular_kpis(sub_sim, parametros)
sub_opt = optimizar_stock_seguridad(sub_forecast, politica, parametros, ss_max=ss_max)
mejor = sub_opt.loc[sub_opt["total_cost"].idxmin()]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Método usado", metodo_usado)
col2.metric("Fill rate", f"{kpis['fill_rate']:.2%}")
col3.metric("Inventario promedio", f"{kpis['avg_inventory']:.1f}")
col4.metric("Ventas perdidas", f"{kpis['lost_sales_units']:.0f}")
col5.metric("Costo total", f"S/ {kpis['total_cost']:,.2f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Mejor método",
    "📊 Datos y pronóstico",
    "📦 Simulación",
    "🎯 Optimización",
    "📋 Tablas",
])

with tab1:
    st.subheader("Mejor método de pronóstico por producto")
    st.write(
        "La app compara Naive, Promedio móvil, SES, Regresión lineal, ARIMA, SARIMA, Holt-Winters y Croston para cada producto. "
        "El mejor método se elige por menor wMAPE. Si hay empate, se toma el Bias más cercano a cero y luego el MAE más bajo."
    )

    resumen_mejores = (
        df_comparacion[df_comparacion["Es mejor"]]
        .copy()
        .sort_values("Producto")
    )

    # IMPORTANTE:
    # df_comparacion ya trae una columna llamada "Mejor método".
    # Por eso NO debemos renombrar "Método" directamente a "Mejor método",
    # porque se crean columnas duplicadas y Streamlit/PyArrow muestra error.
    resumen_mejores = resumen_mejores[["Producto", "Método", "wMAPE", "Bias", "MAE"]].rename(
        columns={"Método": "Mejor método"}
    )

    resumen_mostrar = resumen_mejores.copy()
    resumen_mostrar["wMAPE"] = resumen_mostrar["wMAPE"].map(lambda x: f"{x:.2%}")
    resumen_mostrar["Bias"] = resumen_mostrar["Bias"].map(lambda x: f"{x:.2%}")
    resumen_mostrar["MAE"] = resumen_mostrar["MAE"].map(lambda x: f"{x:,.2f}")

    st.dataframe(resumen_mostrar, use_container_width=True, hide_index=True)

    fig_best = px.bar(
        resumen_mejores,
        x="Producto",
        y="wMAPE",
        color="Mejor método",
        text="Mejor método",
        title="Método ganador por producto según menor wMAPE",
        labels={"wMAPE": "wMAPE", "Producto": "Producto"},
    )
    fig_best.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_best, use_container_width=True)

    csv_mejores = resumen_mejores.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar mejores métodos en CSV",
        data=csv_mejores,
        file_name="mejor_metodo_por_producto.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Pronóstico mensual de demanda")
    st.write(
        "La demanda se trabaja por mes. Si cargaste datos diarios, el sistema los sumó automáticamente por producto y mes. "
        "Además, la app proyecta meses futuros hasta la fecha indicada en el menú lateral."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(grafico_forecast(sub_forecast), use_container_width=True)
    with col_b:
        st.write(f"Comparación de métodos para {producto_sel}")
        st.dataframe(formatear_comparacion(sub_comparacion_producto), use_container_width=True, hide_index=True)
        st.success(
            f"Mejor método para {producto_sel}: {mejor_metodo_producto} "
            f"con wMAPE {mejor_wmape_producto:.2%}."
        )

with tab3:
    st.subheader("Simulación mensual de inventario")
    st.plotly_chart(grafico_inventario(sub_sim), use_container_width=True)

    st.write("KPIs de la simulación")
    kpi_df = pd.DataFrame([kpis]).T.reset_index()
    kpi_df.columns = ["Indicador", "Valor"]
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Optimización de stock de seguridad mensual")
    st.info(
        f"Para el producto {producto_sel}, usando el método de pronóstico {metodo_usado}, "
        f"el stock de seguridad óptimo encontrado es {int(mejor['ss_months'])} meses, "
        f"con costo total aproximado de S/ {mejor['total_cost']:,.2f}."
    )
    st.plotly_chart(grafico_tradeoff(sub_opt), use_container_width=True)

    fig_servicio = px.line(
        sub_opt,
        x="ss_months",
        y="fill_rate",
        markers=True,
        title="Nivel de servicio según meses de stock de seguridad",
        labels={"ss_months": "Meses de stock de seguridad", "fill_rate": "Fill rate"},
    )
    fig_servicio.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_servicio, use_container_width=True)

with tab5:
    st.subheader("Tablas de resultados")
    st.write("Comparación completa de métodos")
    st.dataframe(formatear_comparacion(df_comparacion), use_container_width=True, hide_index=True)

    st.write("Datos mensuales históricos y pronóstico futuro elegido")
    st.dataframe(sub_forecast, use_container_width=True, hide_index=True)

    st.write("Simulación mensual")
    st.dataframe(sub_sim, use_container_width=True, hide_index=True)

    st.write("Resultados de optimización")
    st.dataframe(sub_opt, use_container_width=True, hide_index=True)

    csv = sub_sim.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar simulación mensual en CSV",
        data=csv,
        file_name=f"simulacion_mensual_{producto_sel}.csv",
        mime="text/csv",
    )

    csv_comparacion = df_comparacion.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar comparación de métodos en CSV",
        data=csv_comparacion,
        file_name="comparacion_metodos_pronostico.csv",
        mime="text/csv",
    )
