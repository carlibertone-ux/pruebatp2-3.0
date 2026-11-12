"""
Reentrena y serializa el modelo seleccionado en el TP2 (Gradient Boosting HGB)
para la aplicación del TP4.

Reproduce exactamente el marco experimental del notebook TP2_Regresion_LACT.ipynb:
  - mismas variables (conjunto extendido de 12), mismas variables rezagadas y derivadas
  - partición por vaca 80/20 (GroupShuffleSplit, semilla 42)
  - hiperparámetros elegidos por GroupKFold(5) en el TP2

Uso:
    python entrenar_modelo.py datos/Dataset_Ordenado_Por_Fecha.xlsx

Genera:
    modelo/modelo_l_act.joblib   -> pipeline + metadatos que usa app.py
    datos/historico.csv          -> datos livianos para las visualizaciones de la app
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
TARGET = "L_ACT"
VARS = ["DEL", "NRO_PARTO", "L_ANT", "GR_ANT", "PR_ANT", "GES",
        "DEL_log", "DEL_exp", "MES_sin", "MES_cos", "VERANO", "CS_ANT_log"]
# Hiperparámetros seleccionados en el TP2 (GridSearchCV + GroupKFold(5))
PARAMS_HGB = dict(learning_rate=0.1, max_depth=4, min_samples_leaf=20, l2_regularization=0.0)

COLUMNAS_BASE = ["RP", "NRO_PARTO", "FECHA_PARTO", "DEL", "L_ANT", "L_ACT",
                 "KG GR", "% GR", "KG PR", "% PR", "CS", "GES", "MES"]

RAIZ = Path(__file__).parent


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza, variables rezagadas e ingeniería de variables (idéntico al TP2)."""
    # Solo se usan las columnas originales del control lechero (no las de clasificación/estado sanitario)
    d = df[COLUMNAS_BASE].copy()
    # RP es el identificador de la vaca; en el Excel original es numérico
    rp_num = pd.to_numeric(d["RP"], errors="coerce")
    if rp_num.notna().all():
        d["RP"] = rp_num.astype(int)
    d["FECHA_PARTO"] = pd.to_datetime(d["FECHA_PARTO"])
    d["MES"] = pd.to_datetime(d["MES"])

    # Composición imputada con un valor constante en los primeros controles -> faltante
    sin_comp = d["KG GR"].isna()
    d.loc[sin_comp, ["% GR", "% PR"]] = np.nan

    # Valores del control anterior dentro de la misma lactancia (evita fuga de información)
    d = d.sort_values(["RP", "FECHA_PARTO", "MES"])
    lact = d.groupby(["RP", "FECHA_PARTO"])
    d["GR_ANT"] = lact["% GR"].shift()
    d["PR_ANT"] = lact["% PR"].shift()
    d["CS_ANT_TP2"] = lact["CS"].shift()

    d["DEL_log"] = np.log1p(d["DEL"])
    d["DEL_exp"] = np.exp(-0.05 * d["DEL"])
    mes = d["MES"].dt.month
    d["MES_sin"] = np.sin(2 * np.pi * mes / 12)
    d["MES_cos"] = np.cos(2 * np.pi * mes / 12)
    d["VERANO"] = mes.isin([12, 1, 2, 3]).astype(int)
    d["CS_ANT_log"] = np.log1p(d["CS_ANT_TP2"])

    return d.dropna(subset=["L_ANT", "L_ACT"]).reset_index(drop=True)


def crear_pipeline() -> Pipeline:
    pre = ColumnTransformer([("num", Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler())]), VARS)])
    return Pipeline([("pre", pre),
                     ("modelo", HistGradientBoostingRegressor(random_state=SEMILLA, **PARAMS_HGB))])


def main(ruta_excel: str) -> None:
    df = pd.read_excel(ruta_excel, sheet_name=0)
    d = preparar(df)
    print(f"Filas modeladas: {len(d)} | vacas: {d['RP'].nunique()}")

    X, y, g = d[VARS], d[TARGET], d["RP"]
    idx_ent, idx_pru = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                              random_state=SEMILLA).split(X, y, groups=g))
    modelo = crear_pipeline().fit(X.iloc[idx_ent], y.iloc[idx_ent])

    pred = modelo.predict(X.iloc[idx_pru])
    y_pru = y.iloc[idx_pru].values
    res = y_pru - pred
    metricas = {
        "MAE": mean_absolute_error(y_pru, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_pru, pred))),
        "R2": r2_score(y_pru, pred),
        "n_prueba": int(len(y_pru)),
        "vacas_prueba": int(g.iloc[idx_pru].nunique()),
    }
    print("Métricas en prueba (vacas no vistas):", {k: round(v, 3) for k, v in metricas.items()})

    # Importancia por permutación en prueba (mismo cálculo que en el TP2)
    imp = permutation_importance(modelo, X.iloc[idx_pru], y_pru, n_repeats=10, random_state=SEMILLA,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
    importancia = dict(zip(VARS, imp.importances_mean.round(3).tolist()))

    # Rangos de las variables en entrenamiento: la app los usa para validar y avisar extrapolación
    ent = d.iloc[idx_ent]
    rangos = {v: {"min": float(ent[v].min()), "max": float(ent[v].max()),
                  "p01": float(ent[v].quantile(0.01)), "p99": float(ent[v].quantile(0.99))}
              for v in ["DEL", "NRO_PARTO", "L_ANT", "GR_ANT", "PR_ANT", "GES"]}
    cs = ent["CS_ANT_TP2"]
    rangos["CS_ANT"] = {"min": float(cs.min()), "max": float(cs.max()),
                        "p01": float(cs.quantile(0.01)), "p99": float(cs.quantile(0.99))}

    paquete = {
        "modelo": modelo,
        "variables": VARS,
        "nombre": "Gradient Boosting (HistGradientBoostingRegressor)",
        "hiperparametros": PARAMS_HGB,
        "metricas_prueba": metricas,
        # Cuantiles empíricos del residuo en prueba -> intervalo de predicción del 80 %
        "residuo_q10": float(np.quantile(res, 0.10)),
        "residuo_q90": float(np.quantile(res, 0.90)),
        "residuo_q05": float(np.quantile(res, 0.05)),
        "rangos": rangos,
        "importancia": importancia,
        "version_sklearn": sklearn.__version__,
    }
    (RAIZ / "modelo").mkdir(exist_ok=True)
    joblib.dump(paquete, RAIZ / "modelo" / "modelo_l_act.joblib", compress=3)

    # Datos livianos para los gráficos de contexto de la app
    (RAIZ / "datos").mkdir(exist_ok=True)
    d[["DEL", "NRO_PARTO", "L_ACT"]].to_csv(RAIZ / "datos" / "historico.csv", index=False)
    print("Listo: modelo/modelo_l_act.joblib y datos/historico.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "datos" / "Dataset_Ordenado_Por_Fecha.xlsx"))
