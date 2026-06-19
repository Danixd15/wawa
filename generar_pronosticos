def calcular_meses_futuros(df: pd.DataFrame, fecha_fin) -> tuple[int, pd.Timestamp]:
    """Calcula cuántos meses se deben pronosticar desde el último mes histórico hasta la fecha final."""
    ultima_fecha = pd.to_datetime(df["date"].max()).to_period("M").to_timestamp()
    fecha_fin = pd.to_datetime(fecha_fin).to_period("M").to_timestamp()

    if fecha_fin <= ultima_fecha:
        return 0, ultima_fecha

    meses = (fecha_fin.year - ultima_fecha.year) * 12 + (fecha_fin.month - ultima_fecha.month)
    return int(meses), fecha_fin


def generar_fechas_futuras(ultima_fecha, pasos_futuros: int) -> pd.DatetimeIndex:
    if pasos_futuros <= 0:
        return pd.DatetimeIndex([])
    ultima_fecha = pd.to_datetime(ultima_fecha).to_period("M").to_timestamp()
    return pd.date_range(
        start=ultima_fecha + pd.offsets.MonthBegin(1),
        periods=pasos_futuros,
        freq="MS",
    )


def generar_forecast(df: pd.DataFrame, metodo: str, fecha_fin_pronostico=None) -> pd.DataFrame:
    resultados = []
    pasos_futuros, _ = calcular_meses_futuros(df, fecha_fin_pronostico) if fecha_fin_pronostico is not None else (0, None)

    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)
        pred_hist, pred_future = aplicar_metodo_pronostico(serie, metodo, pasos_futuros)
        err = calcular_errores(serie, pred_hist)

        sub["demand_forecast"] = np.round(pred_hist, 2)
        sub["method_used"] = metodo
        sub["method_wmape"] = err["wMAPE"]
        sub["method_bias"] = err["Bias"]
        sub["tipo_periodo"] = "Histórico"
        resultados.append(sub)

        if pasos_futuros > 0:
            fechas_futuras = generar_fechas_futuras(sub["date"].max(), pasos_futuros)
            futuro = pd.DataFrame(
                {
                    "date": fechas_futuras,
                    "product_id": producto,
                    "demand_real": np.round(pred_future, 2),
                    "demand_forecast": np.round(pred_future, 2),
                    "method_used": metodo,
                    "method_wmape": err["wMAPE"],
                    "method_bias": err["Bias"],
                    "tipo_periodo": "Pronóstico futuro",
                }
            )
            resultados.append(futuro)

    return pd.concat(resultados, ignore_index=True)


def generar_forecast_mejor_por_producto(df: pd.DataFrame, fecha_fin_pronostico=None):
    forecasts_finales = []
    comparacion = []
    pasos_futuros, _ = calcular_meses_futuros(df, fecha_fin_pronostico) if fecha_fin_pronostico is not None else (0, None)

    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)
        predicciones_hist = {}
        predicciones_future = {}
        filas_producto = []

        for metodo in METODOS_PRONOSTICO:
            pred_hist, pred_future = aplicar_metodo_pronostico(serie, metodo, pasos_futuros)
            predicciones_hist[metodo] = pred_hist
            predicciones_future[metodo] = pred_future
            err = calcular_errores(serie, pred_hist)

            fila = {
                "Producto": producto,
                "Método": metodo,
                "wMAPE": err["wMAPE"],
                "Bias": err["Bias"],
                "Abs_Bias": abs(err["Bias"]),
                "MAE": err["MAE"],
            }
            comparacion.append(fila)
            filas_producto.append(fila)

        comp_producto = pd.DataFrame(filas_producto)
        mejor_fila = comp_producto.sort_values(["wMAPE", "Abs_Bias", "MAE"]).iloc[0]
        mejor_metodo = mejor_fila["Método"]

        sub["demand_forecast"] = np.round(predicciones_hist[mejor_metodo], 2)
        sub["method_used"] = mejor_metodo
        sub["method_wmape"] = float(mejor_fila["wMAPE"])
        sub["method_bias"] = float(mejor_fila["Bias"])
        sub["tipo_periodo"] = "Histórico"
        forecasts_finales.append(sub)

        if pasos_futuros > 0:
            fechas_futuras = generar_fechas_futuras(sub["date"].max(), pasos_futuros)
            futuro = pd.DataFrame(
                {
                    "date": fechas_futuras,
                    "product_id": producto,
                    "demand_real": np.round(predicciones_future[mejor_metodo], 2),
                    "demand_forecast": np.round(predicciones_future[mejor_metodo], 2),
                    "method_used": mejor_metodo,
                    "method_wmape": float(mejor_fila["wMAPE"]),
                    "method_bias": float(mejor_fila["Bias"]),
                    "tipo_periodo": "Pronóstico futuro",
                }
            )
            forecasts_finales.append(futuro)

    df_comparacion = pd.DataFrame(comparacion)
    mejores = (
        df_comparacion.sort_values(["Producto", "wMAPE", "Abs_Bias", "MAE"])
        .groupby("Producto", as_index=False)
        .first()[["Producto", "Método"]]
        .rename(columns={"Método": "Mejor método"})
    )

    df_comparacion = df_comparacion.merge(mejores, on="Producto", how="left")
    df_comparacion["Es mejor"] = df_comparacion["Método"] == df_comparacion["Mejor método"]
    df_comparacion = df_comparacion.drop(columns=["Abs_Bias"])

    return pd.concat(forecasts_finales, ignore_index=True), df_comparacion

