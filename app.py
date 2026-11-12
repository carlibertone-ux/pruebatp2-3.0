"""
App de despliegue — TP4
Predicción de la producción diaria de leche por vaca (L_ACT) en el próximo control lechero.
Modelo: Gradient Boosting (HistGradientBoostingRegressor) seleccionado en el TP2.

Ejecutar localmente:  streamlit run app.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).parent
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
AZUL, AZUL_CLARO, GRIS, NARANJA = "#1c5cab", "#86b6ef", "#52514e", "#d9822b"
MIN_DIAS_PARTO_A_PRENEZ = 15   # en el dataset, la gestación siempre es al menos 15 días menor que DEL

st.set_page_config(page_title="Predicción de litros por vaca", page_icon="🐄", layout="wide")


# ---------------------------------------------------------------------------
# Carga del modelo y de los datos de referencia (una sola vez, en caché)
# ---------------------------------------------------------------------------
def buscar_archivo(nombre: str) -> Path:
    """Busca el archivo en su carpeta esperada o en cualquier lugar del repositorio."""
    encontrados = sorted(RAIZ.rglob(nombre), key=lambda p: len(p.parts))
    if not encontrados:
        raise FileNotFoundError(f"No se encontró '{nombre}' en el repositorio. Verificá que esté subido a GitHub.")
    return encontrados[0]


@st.cache_resource
def cargar_modelo():
    return joblib.load(buscar_archivo("modelo_l_act.joblib"))


@st.cache_data
def cargar_historico():
    return pd.read_csv(buscar_archivo("historico.csv"))


try:
    PAQ = cargar_modelo()
    HIST = cargar_historico()
except Exception as e:  # el usuario ve un mensaje claro en lugar de un error de Python
    st.error(f"No se pudo cargar el modelo o los datos de referencia: {e}")
    st.stop()

MODELO, VARS, RANGOS = PAQ["modelo"], PAQ["variables"], PAQ["rangos"]
MET = PAQ["metricas_prueba"]


def n(x, dec=1):
    """Formatea un número al estilo argentino: coma decimal y punto de miles."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def construir_fila(del_dias, nro_parto, l_ant, gr_ant, pr_ant, ges, mes, cs_ant):
    """Arma la fila de entrada con las mismas variables derivadas que se usaron al entrenar."""
    fila = {
        "DEL": del_dias, "NRO_PARTO": nro_parto, "L_ANT": l_ant,
        "GR_ANT": gr_ant if gr_ant is not None else np.nan,
        "PR_ANT": pr_ant if pr_ant is not None else np.nan,
        "GES": ges,
        "DEL_log": np.log1p(del_dias), "DEL_exp": np.exp(-0.05 * del_dias),
        "MES_sin": np.sin(2 * np.pi * mes / 12), "MES_cos": np.cos(2 * np.pi * mes / 12),
        "VERANO": int(mes in (12, 1, 2, 3)),
        "CS_ANT_log": np.log1p(cs_ant) if cs_ant is not None else np.nan,
    }
    return pd.DataFrame([fila])[VARS]


def fuera_de_lo_habitual(nombre, valor, etiqueta):
    """Devuelve un aviso si el valor está fuera del 98 % central de los datos de entrenamiento."""
    r = RANGOS[nombre]
    if valor is None:
        return None
    if valor < r["p01"] or valor > r["p99"]:
        return (f"**{etiqueta} = {valor:g}** está fuera de los valores habituales del tambo "
                f"({r['p01']:g} a {r['p99']:g}). El modelo vio pocos casos así y la estimación es menos confiable.")
    return None


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🐄 ¿Cuántos litros dará la vaca en el próximo control?")
st.markdown(
    "Esta herramienta estima la **producción diaria de leche (litros/día)** de una vaca en su próximo "
    "control lechero, a partir de lo que el tambo ya sabe **antes** del control: los litros del control "
    "anterior, la etapa de lactancia, el número de parto, la preñez y la época del año. "
    "Sirve para planificar la remisión de leche y para **detectar caídas de producción inesperadas**, "
    "que pueden ser una señal temprana de mastitis, problemas de alimentación o estrés calórico."
)
st.caption(
    f"Modelo: *Gradient Boosting* entrenado con {n(len(HIST), 0)} controles de un tambo de Rafaela (Santa Fe). "
    f"Error medio en vacas que el modelo nunca vio: ±{n(MET['MAE'])} L/día."
)

col_entrada, col_salida = st.columns([1, 1.25], gap="large")

# ---------------------------------------------------------------------------
# Entradas del usuario (con rangos válidos tomados de los datos de entrenamiento)
# ---------------------------------------------------------------------------
with col_entrada:
    st.subheader("1. Datos de la vaca")

    c1, c2 = st.columns(2)
    nro_parto = c1.selectbox("Número de parto", list(range(1, int(RANGOS["NRO_PARTO"]["max"]) + 1)), index=1,
                             help="1 = vaquillona de primer parto (primípara).")
    del_dias = c2.number_input("Días en leche (DEL)", min_value=1, max_value=int(RANGOS["DEL"]["max"]),
                               value=65, step=1, help="Días transcurridos desde el último parto.")

    l_ant = st.number_input("Litros/día en el control anterior", min_value=float(RANGOS["L_ANT"]["min"]),
                            max_value=float(RANGOS["L_ANT"]["max"]), value=28.5, step=0.1, format="%.1f",
                            help="Es la variable que más pesa en la predicción.")

    mes_nombre = st.selectbox("Mes del próximo control", MESES, index=date.today().month - 1)
    mes = MESES.index(mes_nombre) + 1

    st.subheader("2. Reproducción")
    prenada = st.radio("¿Está preñada?", ["No", "Sí"], horizontal=True)
    ges = 0
    if prenada == "Sí":
        ges = st.number_input("Días de gestación", min_value=1, max_value=int(RANGOS["GES"]["max"]),
                              value=min(60, int(RANGOS["GES"]["max"])), step=1)

    st.subheader("3. Calidad del control anterior (opcional)")
    st.caption("Si no tenés el dato, el modelo usa el valor típico del tambo.")
    tiene_comp = st.checkbox("Tengo grasa y proteína del control anterior", value=True)
    gr_ant = pr_ant = None
    if tiene_comp:
        c3, c4 = st.columns(2)
        gr_ant = c3.number_input("% grasa", min_value=float(RANGOS["GR_ANT"]["min"]),
                                 max_value=float(RANGOS["GR_ANT"]["max"]), value=3.65, step=0.01, format="%.2f")
        pr_ant = c4.number_input("% proteína", min_value=float(RANGOS["PR_ANT"]["min"]),
                                 max_value=float(RANGOS["PR_ANT"]["max"]), value=3.35, step=0.01, format="%.2f")
    tiene_cs = st.checkbox("Tengo el recuento de células somáticas del control anterior", value=True)
    cs_ant = None
    if tiene_cs:
        cs_ant = st.number_input("Células somáticas (miles/ml)", min_value=1, max_value=int(RANGOS["CS_ANT"]["max"]),
                                 value=120, step=10, help="Por encima de 200 mil/ml suele indicar mastitis subclínica.")

# ---------------------------------------------------------------------------
# Validación lógica de las entradas
# ---------------------------------------------------------------------------
errores, avisos = [], []
if ges > 0 and ges > del_dias - MIN_DIAS_PARTO_A_PRENEZ:
    errores.append(f"Los días de gestación ({ges}) no pueden ser mayores que los días en leche menos "
                   f"{MIN_DIAS_PARTO_A_PRENEZ} ({del_dias - MIN_DIAS_PARTO_A_PRENEZ}): la vaca quedó preñada "
                   "después del parto. Revisá ambos valores.")
for nombre, valor, etiqueta in [("DEL", del_dias, "Días en leche"), ("L_ANT", l_ant, "Litros anteriores"),
                                ("GR_ANT", gr_ant, "% grasa"), ("PR_ANT", pr_ant, "% proteína"),
                                ("NRO_PARTO", nro_parto, "Número de parto"), ("CS_ANT", cs_ant, "Células somáticas")]:
    a = fuera_de_lo_habitual(nombre, valor, etiqueta)
    if a:
        avisos.append(a)
if cs_ant is not None and cs_ant > 200:
    avisos.append("El recuento de células somáticas anterior supera 200 mil/ml (posible mastitis subclínica). "
                  "Conviene revisar el estado sanitario de la ubre.")

# ---------------------------------------------------------------------------
# Predicción y resultado
# ---------------------------------------------------------------------------
with col_salida:
    st.subheader("Resultado")
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    pred = float(MODELO.predict(construir_fila(del_dias, nro_parto, l_ant, gr_ant, pr_ant, ges, mes, cs_ant))[0])
    lo, hi = max(0.0, pred + PAQ["residuo_q10"]), pred + PAQ["residuo_q90"]
    cambio = pred - l_ant

    m1, m2 = st.columns(2)
    m1.metric("Producción estimada", f"{n(pred)} L/día",
              delta=f"{'+' if cambio >= 0 else '−'}{n(abs(cambio))} L vs. control anterior")
    m2.metric("Rango probable (80 %)", f"{n(lo)} – {n(hi)} L")
    st.markdown(
        f"Para una vaca de **{nro_parto}.º parto** con **{del_dias} días en leche** que dio **{n(l_ant)} L** "
        f"en el control anterior, el modelo espera unos **{n(pred)} litros/día** en {mes_nombre.lower()}. "
        f"En 8 de cada 10 casos el valor real cae entre {n(lo)} y {n(hi)} L."
    )
    for a in avisos:
        st.warning(a)

    # --- Uso práctico: comparar contra el dato real una vez hecho el control ---
    with st.expander("🔎 ¿Ya tenés el resultado real del control? Compará contra lo esperado"):
        real = st.number_input("Litros/día medidos en el control", min_value=0.0, max_value=80.0,
                               value=None, step=0.1, format="%.1f", placeholder="Ingresá el valor real")
        if real is not None:
            dif = real - pred
            if dif < PAQ["residuo_q05"]:
                st.error(f"Dio {n(abs(dif))} L **menos** de lo esperado: es una caída mayor que la del 95 % de las "
                         "vacas en situación similar. Conviene revisar sanidad de la ubre, alimentación o estrés calórico.")
            elif dif < PAQ["residuo_q10"]:
                st.warning(f"Dio {n(abs(dif))} L menos de lo esperado: algo por debajo de lo habitual. "
                           "Conviene seguirla de cerca en el próximo control.")
            elif dif > PAQ["residuo_q90"]:
                st.success(f"Dio {n(dif)} L **más** de lo esperado: rinde por encima de lo habitual.")
            else:
                st.info(f"Diferencia de {'+' if dif >= 0 else '−'}{n(abs(dif))} L: dentro de lo esperable para el modelo.")

    # -----------------------------------------------------------------------
    # Contexto para interpretar el resultado
    # -----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Comparación con el tambo", "Curva de lactancia", "Sobre el modelo"])

    with tab1:
        percentil = (HIST["L_ACT"] < pred).mean() * 100
        fig = go.Figure()
        fig.add_histogram(x=HIST["L_ACT"], xbins=dict(size=1), marker_color=AZUL_CLARO, name="Controles históricos")
        fig.add_vrect(x0=lo, x1=hi, fillcolor=AZUL, opacity=0.12, line_width=0)
        fig.add_vline(x=pred, line_color=AZUL, line_width=3,
                      annotation_text=f"Predicción: {n(pred)} L", annotation_position="top")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          xaxis_title="Litros/día", yaxis_title="Cantidad de controles", bargap=0.05)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"La producción estimada supera al {percentil:.0f} % de los controles registrados en el tambo. "
                   "La franja sombreada es el rango probable.")

    with tab2:
        h = HIST[HIST["DEL"] <= 600].copy()
        h["grupo"] = np.where(h["NRO_PARTO"] == 1, "Primíparas (1.er parto)", "Multíparas (2+ partos)")
        h["tramo"] = (h["DEL"] // 20) * 20 + 10
        curva = h.groupby(["grupo", "tramo"])["L_ACT"].median().reset_index()
        fig2 = go.Figure()
        for grupo, color in [("Primíparas (1.er parto)", NARANJA), ("Multíparas (2+ partos)", AZUL)]:
            c = curva[curva["grupo"] == grupo]
            fig2.add_scatter(x=c["tramo"], y=c["L_ACT"], mode="lines", name=grupo,
                             line=dict(color=color, width=2.5))
        fig2.add_scatter(x=[del_dias], y=[pred], mode="markers", name="Esta vaca (predicción)",
                         marker=dict(color="black", size=13, symbol="star"),
                         error_y=dict(type="data", symmetric=False, array=[hi - pred], arrayminus=[pred - lo],
                                      color=GRIS))
        fig2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                           xaxis_title="Días en leche", yaxis_title="Litros/día (mediana)",
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Curvas de lactancia típicas del tambo (mediana por tramos de 20 días). "
                   "Permite ver si la vaca está en el pico, en el descenso o al final de la lactancia.")

    with tab3:
        st.markdown(
            f"""
**Algoritmo:** *Gradient Boosting* (`HistGradientBoostingRegressor` de *scikit-learn*), elegido en el TP2
entre 6 modelos (línea de base, regresión lineal, *Ridge* polinómica, árbol de decisión, *Random Forest* y
*Gradient Boosting*) por tener el menor error en validación cruzada agrupada por vaca.

**Desempeño en vacas no vistas** ({MET['vacas_prueba']} vacas, {n(MET['n_prueba'], 0)} controles):

| Métrica | Valor |
|---|---|
| Error absoluto medio (MAE) | {n(MET['MAE'], 2)} L/día |
| Raíz del error cuadrático medio (RMSE) | {n(MET['RMSE'], 2)} L/día |
| Coeficiente de determinación (R²) | {n(MET['R2'], 3)} |
"""
        )
        grupos = {
            "Litros del control anterior": ["L_ANT"],
            "Etapa de lactancia": ["DEL", "DEL_log", "DEL_exp"],
            "Época del año": ["MES_sin", "MES_cos", "VERANO"],
            "Número de parto": ["NRO_PARTO"],
            "Preñez": ["GES"],
            "Grasa y proteína anteriores": ["GR_ANT", "PR_ANT"],
            "Células somáticas anteriores": ["CS_ANT_log"],
        }
        imp = pd.Series({g: max(0.0, sum(PAQ["importancia"][v] for v in vs)) for g, vs in grupos.items()})
        imp = imp.sort_values()
        fig3 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=AZUL))
        fig3.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Aumento del error (L) si se desordena la variable")
        st.markdown("**¿Qué datos pesan más en la predicción?**")
        st.plotly_chart(fig3, width="stretch")
        st.markdown(
            """
**Limitaciones a tener en cuenta**
- Entrenado con datos de **un solo tambo** de Rafaela: en otros establecimientos (otra genética, otra dieta) puede equivocarse más.
- Necesita los litros del control anterior, así que **no sirve para el primer control** después del parto.
- No conoce eventos nuevos entre controles (enfermedad, cambio de ración, ola de calor puntual).
- Es una herramienta de apoyo: no reemplaza el diagnóstico veterinario.
"""
        )

st.divider()
st.caption("Trabajo Práctico 4 — Inteligencia Artificial y Aprendizaje Automático — UNRaf, 2026.")
