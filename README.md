# Certamen 1 ML · Ensambles, costo de clasificación y umbral óptimo

**Grupo 15** · Proyecto grupal (UDD · Machine Learning). Compara 5 modelos × 2 estrategias de desbalance (SMOTE / pesos), elige el mejor por F1 en CV y traduce sus errores a dinero para encontrar el umbral de decisión que minimiza el costo, con un dashboard interactivo en Streamlit.

**Base de datos:** [Health Insurance Cross Sell Prediction (Kaggle)](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) · 381.109 registros · 10 predictoras · 12,26 % de clase minoritaria (`Response`).

## Contenido

| Archivo / carpeta | Descripción |
|---|---|
| `Certamen1_Grupo_15.ipynb` | Notebook completo, ejecutado de principio a fin (semilla 42) |
| `app.py` | Dashboard Streamlit (4 pestañas: costo vs umbral, desglose del costo, ROC con iso-costo, guía) |
| `.streamlit/config.toml` | Tema visual del dashboard |
| `dashboard_data/predicciones_test.csv` | `y_test` y probabilidades en escala real, usadas por el dashboard |
| `models/` | Mejor modelo reentrenado (`joblib`) y metadatos |
| `resultados/` | Tablas (CSV) y gráficos generados por el notebook |
| `requirements.txt` | Dependencias del dashboard |
| `data/` | *(no versionada)* poner aquí `train.csv` descargado de Kaggle |

## Cómo reproducir

1. Descargar la base de Kaggle y dejar `train.csv` en `data/` (en Colab, el notebook pide subirlo).
2. `pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm plotly seaborn matplotlib joblib streamlit`
3. Ejecutar el notebook (≈ 12 min en una CPU de 12 núcleos).
4. Dashboard: `streamlit run app.py`

## Resultados principales (test, C_FP = US$ 5, C_FN = US$ 50)

- Mejor modelo por CV: **Random Forest con pesos** (F1 CV 0,439; F1 test 0,444).
- Umbral 0,5 ≈ no hacer nada (costo US$ 462.565). Umbral teórico 0,091 y empírico 0,085 bajan el costo a ≈ US$ 142.000 (**ahorro ≈ US$ 320.000, 69 %**).

Los costos de FP y FN son supuestos de negocio declarados en el notebook y editables en el dashboard.
