# Predicción de litros por vaca en el próximo control lechero

Aplicación web del **Trabajo Práctico 4** de *Inteligencia Artificial y Aprendizaje Automático* (UNRaf, 2026).

Estima la **producción diaria de leche (L/día)** que dará una vaca en su próximo control lechero, a partir de datos conocidos **antes** del control: litros del control anterior, días en leche, número de parto, días de gestación, mes del año y, de forma opcional, grasa, proteína y células somáticas del control anterior.

🔗 **App en línea:** _(pegar aquí la URL de Streamlit Community Cloud)_

## Modelo

| | |
|---|---|
| Algoritmo | *Gradient Boosting* (`HistGradientBoostingRegressor`, *scikit-learn*) |
| Hiperparámetros | `learning_rate=0.1`, `max_depth=4`, `min_samples_leaf=20`, `l2_regularization=0` |
| Selección | Menor RMSE en validación cruzada `GroupKFold(5)` agrupada por vaca, entre 6 modelos (TP2) |
| Datos | 5.993 controles de 305 vacas de un tambo de Rafaela (Santa Fe) |
| Desempeño en prueba (61 vacas no vistas) | MAE = 2,87 L/día · RMSE = 3,81 L/día · R² = 0,668 |

El modelo se guarda como un `Pipeline` completo (imputación por mediana + estandarización + modelo), así que la app no necesita reentrenar ni repetir el preprocesamiento a mano.

## Qué hace la app

- **Entrada de datos** con selectores y campos numéricos limitados a los rangos observados en el tambo.
- **Validación**: bloquea combinaciones imposibles (días de gestación mayores que los días en leche) y avisa cuando un valor está fuera del 98 % central de los datos de entrenamiento (la predicción es menos confiable).
- **Resultado**: litros estimados, cambio respecto del control anterior y rango probable del 80 % (calculado con los cuantiles de los residuos en el conjunto de prueba).
- **Comparación con el dato real**: una vez hecho el control, indica si la caída es mayor a la esperable (posible alerta sanitaria o de manejo).
- **Contexto**: ubicación de la predicción en la distribución histórica del tambo, curva de lactancia típica (primíparas vs. multíparas) y peso de cada dato en la predicción.

## Estructura del repositorio

```
├── app.py                      # Aplicación Streamlit
├── entrenar_modelo.py          # Reentrena y serializa el modelo (reproduce el TP2)
├── modelo/
│   └── modelo_l_act.joblib     # Pipeline entrenado + métricas y rangos válidos
├── datos/
│   ├── Dataset_Ordenado_Por_Fecha.xlsx   # Dataset del proyecto
│   └── historico.csv           # Extracto liviano usado por los gráficos de la app
├── requirements.txt
└── .streamlit/config.toml      # Tema visual
```

## Ejecución local

Requiere Python 3.13 (las versiones de `requirements.txt` están fijadas para que el modelo serializado se cargue sin errores).

```bash
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

### Reentrenar el modelo (opcional)

```bash
python entrenar_modelo.py datos/Dataset_Ordenado_Por_Fecha.xlsx
```

Regenera `modelo/modelo_l_act.joblib` y `datos/historico.csv`. Con la semilla 42 se obtienen exactamente las métricas del TP2.

## Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub (público).
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub → **Create app**.
3. Elegir el repositorio, la rama `main` y el archivo `app.py`.
4. En **Advanced settings**, elegir **Python 3.13**.
5. **Deploy**. Copiar la URL pública en este README.

## Limitaciones

- Entrenado con datos de un solo tambo: en otros establecimientos el error puede ser mayor.
- Requiere los litros del control anterior, por lo que no aplica al primer control de la lactancia.
- No contempla eventos ocurridos entre controles (enfermedades, cambios de ración, olas de calor).
- Es una herramienta de apoyo a la decisión; no reemplaza el diagnóstico veterinario.

## Autores

_(Nombres del grupo)_ — Universidad Nacional de Rafaela, 2026.
