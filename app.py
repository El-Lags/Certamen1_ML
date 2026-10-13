"""Dashboard interactivo: costo de clasificación y umbral óptimo (Certamen 1 ML).

Ejecutar:  streamlit run app.py
Lee dashboard_data/predicciones_test.csv (y_test y probabilidades en escala real, generadas en el notebook).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

st.set_page_config(page_title="Costo y umbral óptimo", layout="wide")

# Paleta definida una sola vez: cada concepto siempre tiene el mismo color en todos los gráficos
NAVY, RED, TEAL, GREY, ORANGE = "#1f3a5f", "#d1495b", "#2a9d8f", "#9aa5b1", "#e08e45"

# Estilo: tarjetas con borde para los indicadores, título con barra lateral y cuadro de "lectura rápida"
st.markdown(f"""
<style>
div[data-testid="stMetric"] {{background:#F5F7FA; border:1px solid #E1E6EE; border-radius:8px; padding:14px 16px;}}
div[data-testid="stMetric"] label p {{color:#5F6B7A; font-size:0.85rem;}}
.titulo {{border-left:5px solid {NAVY}; padding-left:16px; margin-bottom:6px;}}
.titulo h1 {{margin:0; padding:0; font-size:2rem; color:#1B2430;}}
.subtitulo {{color:#5F6B7A; font-size:0.95rem; margin:4px 0 18px 21px;}}
.lectura {{background:#F5F7FA; border:1px solid #E1E6EE; border-left:5px solid {TEAL}; border-radius:8px;
          padding:14px 18px; margin:18px 0 8px 0; color:#1B2430;}}
.lectura b {{color:{NAVY};}}
</style>""", unsafe_allow_html=True)


def usd(x):
    return f"${x:,.0f}"


@st.cache_data
def cargar():
    """Carga las probabilidades del test y precalcula FP y FN para cada umbral (no dependen de los costos)."""
    d = pd.read_csv("dashboard_data/predicciones_test.csv")
    y, p = d["y_test"].to_numpy(), d["proba"].to_numpy()
    grid = np.round(np.linspace(0.0, 1.0, 501), 4)
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(pos, grid, side="left")             # positivos con proba < umbral
    fp = len(neg) - np.searchsorted(neg, grid, side="left")  # negativos con proba >= umbral
    fpr, tpr, _ = roc_curve(y, p)
    return y, p, grid, fp, fn, fpr, tpr


y, p, grid, FP, FN, FPR, TPR = cargar()
P, N = int((y == 1).sum()), int((y == 0).sum())

st.markdown('<div class="titulo"><h1>Venta cruzada de seguros: costo de cada error y umbral óptimo</h1></div>'
            f'<div class="subtitulo">Health Insurance Cross Sell · {len(y):,} clientes de test · {P:,} compradores ({P / len(y):.1%}) · '
            'Random Forest con probabilidades en escala real</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- controles (barra lateral)
with st.sidebar:
    st.header("Panel de control")
    st.subheader("1. Costos de los errores (US$)")
    c_fp = st.slider("Falso positivo: contactar a quien NO compra", 0.5, 50.0, 5.0, 0.5)
    c_fn = st.slider("Falso negativo: perder una venta", 1.0, 500.0, 50.0, 1.0)
    p_star = c_fp / (c_fp + c_fn)
    st.metric("Umbral teórico p* = C_FP / (C_FP + C_FN)", f"{p_star:.3f}", help=f"Razón FN/FP = {c_fn / c_fp:.1f}")
    st.subheader("2. Umbral de decisión")
    modo = st.radio("Umbral a evaluar", ["Óptimo (mínimo costo)", "Teórico p*", "Manual"], index=0)
    with st.expander("Umbral manual (avanzado)", expanded=(modo == "Manual")):
        manual = st.slider("Umbral manual", 0.0, 1.0, 0.5, 0.01, disabled=(modo != "Manual"),
                           help="Se activa al elegir 'Manual' arriba.")

# ---------------------------------------------------------------- cálculos (todo vectorizado)
costo = FP * c_fp + FN * c_fn                      # costo total para cada umbral de la grilla
i_opt = int(np.argmin(costo))
u_opt = float(grid[i_opt])


def idx(u):
    return int(np.argmin(np.abs(grid - u)))


u_sel = {"Óptimo (mínimo costo)": u_opt, "Teórico p*": p_star, "Manual": manual}[modo]
i_sel, i_05, i_ps = idx(u_sel), idx(0.5), idx(p_star)

tp, tn = P - FN[i_sel], N - FP[i_sel]
recall = tp / P
prec = tp / (tp + FP[i_sel]) if (tp + FP[i_sel]) else 0.0
ahorro = costo[i_05] - costo[i_sel]

# ---------------------------------------------------------------- indicadores (siempre visibles)
k = st.columns(5)
k[0].metric("Umbral evaluado", f"{u_sel:.3f}", help=f"Óptimo en estos datos: {u_opt:.3f} · teórico: {p_star:.3f}")
k[1].metric("Recall", f"{recall:.1%}", help="% de compradores que sí contactamos")
k[2].metric("Precisión", f"{prec:.1%}", help="% de contactados que sí compran")
k[3].metric("Costo total", usd(costo[i_sel]))
k[4].metric("Ahorro vs umbral 0,5", usd(ahorro), delta=f"{ahorro / costo[i_05]:.1%}" if costo[i_05] else None)

# Lectura rápida: texto que se arma solo con los valores actuales
st.markdown(
    f'<div class="lectura"><b>Lectura rápida.</b> Con un falso positivo de <b>{usd(c_fp)}</b> y un falso negativo de <b>{usd(c_fn)}</b> '
    f'(razón {c_fn / c_fp:.1f}), el umbral teórico es <b>{p_star:.3f}</b> y el óptimo en estos datos es <b>{u_opt:.3f}</b>. '
    f'Con el umbral evaluado ({u_sel:.3f}) se contacta al <b>{recall:.1%}</b> de los compradores y el costo total es '
    f'<b>{usd(costo[i_sel])}</b>, frente a <b>{usd(costo[i_05])}</b> con el umbral 0,5: '
    + (f'un ahorro de <b>{usd(ahorro)}</b> ({ahorro / costo[i_05]:.1%}).' if ahorro >= 0 else f'<b>{usd(-ahorro)}</b> más caro que con 0,5.')
    + '</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Costo vs umbral", "¿En qué se va la plata?", "Curva ROC", "Cómo leerlo"])

# ---------------------------------------------------------------- pestaña 1: curva de costo + matriz de confusión
with tab1:
    col1, col2 = st.columns([1.6, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=costo, mode="lines", line=dict(color=NAVY, width=3), name="Costo total",
                                 hovertemplate="umbral %{x:.3f}<br>costo $%{y:,.0f}<extra></extra>"))
        for u, c, n, pos_ in [(0.5, GREY, "Umbral 0,5", "top"), (u_opt, RED, f"Óptimo {u_opt:.3f}", "top")]:
            fig.add_vline(x=u, line=dict(color=c, dash="dash"), annotation_text=n, annotation_position=pos_)
        fig.add_vline(x=p_star, line=dict(color=TEAL, dash="dot"), annotation_text=f"p* {p_star:.3f}", annotation_position="bottom")
        fig.add_trace(go.Scatter(x=[u_sel], y=[costo[i_sel]], mode="markers", name="Umbral evaluado",
                                 marker=dict(size=13, color=RED, line=dict(width=2, color="white"))))
        fig.update_layout(title="Costo total vs umbral (test)", xaxis_title="Umbral", yaxis_title="Costo total (US$)",
                          height=430, margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with col2:
        txt = [[f"<b>TN</b><br>{tn:,}<br>$0", f"<b>FP</b><br>{FP[i_sel]:,}<br>{usd(FP[i_sel] * c_fp)}"],
               [f"<b>FN</b><br>{FN[i_sel]:,}<br>{usd(FN[i_sel] * c_fn)}", f"<b>TP</b><br>{tp:,}<br>$0"]]
        cm = go.Figure(go.Heatmap(z=[[0, 1], [1, 0]], text=txt, texttemplate="%{text}",
                                  colorscale=[[0, "#dfe8f3"], [1, "#f4c9cf"]], showscale=False,
                                  x=["Pred. 0", "Pred. 1"], y=["Real 0", "Real 1"], hoverinfo="skip"))
        cm.update_layout(title=f"Matriz de confusión @ {u_sel:.3f} (cantidad y costo)", height=430,
                         margin=dict(t=60, b=40), yaxis_autorange="reversed")
        st.plotly_chart(cm, width="stretch")

# ---------------------------------------------------------------- pestaña 2: desglose del costo + comparación
with tab2:
    est = pd.DataFrame({
        "Estrategia": ["No hacer nada", "Contactar a todos", "Umbral 0,5", f"Teórico {p_star:.3f}",
                       f"Óptimo {u_opt:.3f}", f"Evaluado {u_sel:.3f}"],
        "Costo FP": [0, N * c_fp, FP[i_05] * c_fp, FP[i_ps] * c_fp, FP[i_opt] * c_fp, FP[i_sel] * c_fp],
        "Costo FN": [P * c_fn, 0, FN[i_05] * c_fn, FN[i_ps] * c_fn, FN[i_opt] * c_fn, FN[i_sel] * c_fn]})
    est["Costo total"] = est["Costo FP"] + est["Costo FN"]
    est["Ahorro vs 0,5"] = costo[i_05] - est["Costo total"]

    col3, col4 = st.columns([1.3, 1])
    with col3:
        o = est.sort_values("Costo total")
        bar = go.Figure()
        bar.add_trace(go.Bar(y=o["Estrategia"], x=o["Costo FP"], orientation="h", name=f"FP × {usd(c_fp)}", marker_color=NAVY))
        bar.add_trace(go.Bar(y=o["Estrategia"], x=o["Costo FN"], orientation="h", name=f"FN × {usd(c_fn)}", marker_color=ORANGE))
        bar.update_layout(barmode="stack", title="Costo por estrategia: contactos inútiles (FP) vs ventas perdidas (FN)",
                          height=400, margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2),
                          yaxis=dict(autorange="reversed"), xaxis_title="US$")
        st.plotly_chart(bar, width="stretch")
    with col4:
        st.subheader("Comparación de estrategias")
        st.dataframe(est.style.format({c: "${:,.0f}" for c in ["Costo FP", "Costo FN", "Costo total", "Ahorro vs 0,5"]}),
                     hide_index=True, width="stretch")
        st.download_button("Descargar tabla (CSV)", est.to_csv(index=False).encode("utf-8"),
                           "comparacion_estrategias.csv", "text/csv")

# ---------------------------------------------------------------- pestaña 3: ROC con recta de iso-costo
with tab3:
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=FPR, y=TPR, mode="lines", line=dict(color=NAVY, width=3), name="ROC"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dot"), name="Azar"))
    pendiente = (N * c_fp) / (P * c_fn)                      # pendiente de la recta de iso-costo
    f_sel, t_sel = FP[i_sel] / N, tp / P
    xs = np.linspace(0, 1, 50)
    roc.add_trace(go.Scatter(x=xs, y=np.clip(t_sel + pendiente * (xs - f_sel), 0, 1), mode="lines",
                             line=dict(color=RED, dash="dash"), name="Iso-costo del umbral evaluado"))
    for i, n, c in [(i_05, "Umbral 0,5", GREY), (i_ps, f"Teórico {p_star:.3f}", TEAL)]:
        roc.add_trace(go.Scatter(x=[FP[i] / N], y=[(P - FN[i]) / P], mode="markers", name=n, marker=dict(size=11, color=c)))
    roc.add_trace(go.Scatter(x=[f_sel], y=[t_sel], mode="markers", name="Umbral evaluado",
                             marker=dict(size=14, color=RED, line=dict(width=2, color="white"))))
    roc.update_layout(title="Curva ROC con recta de iso-costo", xaxis_title="Tasa de FP (FPR)", yaxis_title="Recall (TPR)",
                      height=480, margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2), yaxis_range=[0, 1.02])
    st.plotly_chart(roc, width="stretch")
    st.caption("Todos los puntos sobre una misma recta de iso-costo cuestan lo mismo. El mejor umbral es donde la recta "
               "toca la ROC más arriba y a la izquierda; al encarecer el FN la recta se aplana y el óptimo sube por la curva.")

# ---------------------------------------------------------------- pestaña 4: explicación
with tab4:
    st.markdown(f"""
**Problema.** Una aseguradora quiere ofrecer un seguro de vehículo a sus clientes de salud. Contactar a alguien cuesta dinero
(**falso positivo**), pero no contactar a alguien que habría comprado es una venta perdida (**falso negativo**).

**Umbral teórico.** Se contacta a un cliente si su probabilidad de compra es mayor o igual a
`p* = C_FP / (C_FP + C_FN)` = {c_fp:g} / ({c_fp:g} + {c_fn:g}) = **{p_star:.3f}**. Con el umbral 0,5 se ignora que un FN cuesta
{c_fn / c_fp:.1f} veces lo que un FP.

**Qué hacer en el dashboard.**
1. Cambia los costos en la barra lateral y mira cómo se mueve p* y el óptimo.
2. Elige *Manual* y arrastra el umbral para compararlo con el óptimo.
3. Revisa en qué se va la plata (pestaña 2) y dónde cae cada umbral sobre la ROC (pestaña 3).

**Advertencias.** El umbral óptimo se calcula aquí sobre el test solo para visualizarlo; en el notebook el umbral empírico se elige
con el train (`TunedThresholdClassifierCV`). Los costos son supuestos de negocio editables. El modelo usa género y edad, por lo que
conviene monitorear que no trate distinto a ningún grupo.
""")
