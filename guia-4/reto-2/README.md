# Reto 2 — Clasificación de cobertura terrestre (landuse)

Comparación de tres enfoques de aprendizaje profundo sobre el dataset local `landuse-dataset/`
(3 clases: `denseresidential`, `harbor`, `river`; 500 imágenes por clase):

| Grupo | Modelo | Cuaderno | Resultados |
|---|---|---|---|
| CNN | CNN base (8-16-32) | `LandUseCNN.ipynb` | `resultados/cnn/base/` |
| CNN | CNN mejorada (32-64-128 + BN + Dropout) | `LandUseCNN.ipynb` | `resultados/cnn/mejorada/` |
| DL no secuencial | DenseNet121 preentrenada + fine-tuning | `LandUseDenseNet.ipynb` | `resultados/densenet/` |
| Transformer | Swin desde cero | `LandUseSWIN.ipynb` | `resultados/swin/` |

## Protocolo

- Split común: 75 % train, 15 % validación y 25 % test, estratificado, semilla 20
  (`resultados/split.json`).
- Imágenes redimensionadas a 128×128.
- Métricas: accuracy, precisión/recall/F1 macro y weighted, Cohen's kappa,
  ROC-AUC one-vs-rest, matriz de confusión, IC 95 % por bootstrap y test exacto
  de McNemar.
- Todos los modelos se cargan desde `resultados/` por defecto. Para reentrenar,
  cambia `FORCE_RETRAIN = True` en la celda de configuración del cuaderno.
- Cada carpeta de modelo guarda `metrics.json`, `predictions.csv`,
  `classification_report.txt`, `history.csv` y `model.keras`.
- En `resultados/swin/` se guarda además `model.weights.h5`: el `.keras` contiene
  capas personalizadas (atención por ventanas), así que los pesos se conservan
  por separado para poder reconstruir y recargar el modelo.
- `reto-2.ipynb` genera las **4 figuras** de la comparación: barras con IC 95 %,
  matrices de confusión, curvas de aprendizaje y heatmap de F1 por clase.

## Resumen de resultados

| Modelo | Accuracy | F1 macro | Parámetros | Entrenamiento |
|---|---:|---:|---:|---:|
| CNN base | 0.8773 | 0.8755 | 30 611 | 76 s |
| CNN mejorada | 0.9760 | 0.9760 | 305 251 | 2063 s |
| **DenseNet121** | **0.9893** | **0.9893** | 7 040 579 | 698 s |
| Swin Transformer | 0.9653 | 0.9654 | 5 133 871 | 833 s |

El análisis completo, las figuras y las conclusiones están en `reto-2.ipynb` y
en `resultados/comparacion/`.

## Estructura de `resultados/`

```text
resultados/
├── split.json
├── cnn/base/
├── cnn/mejorada/
├── densenet/            # incluye gradcam.png
├── swin/                # incluye model.weights.h5
└── comparacion/
    ├── model_summary.csv
    ├── mcnemar_pvalues.csv
    ├── barras_accuracy_f1.png
    ├── matrices_confusion.png
    ├── curvas_aprendizaje.png
    └── heatmap_f1_por_clase.png
```

## Dependencias

Además de `numpy`, `opencv-python` y `scipy`, estos cuadernos usan
`tensorflow`/`keras`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`,
`jupyter`/`nbconvert`. Están instalados en el entorno `venv` del repositorio.

## Reproducción

Desde esta carpeta y con el entorno virtual del repositorio:

```bash
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace LandUseCNN.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace LandUseDenseNet.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace LandUseSWIN.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace reto-2.ipynb
```

`common_landuse.py` centraliza la carga de imágenes, el split, las métricas, el
guardado de resultados y la lógica de **cargar o reentrenar** para que todos los
cuadernos usen exactamente el mismo protocolo.
