# Reto 2 — Clasificación de cobertura terrestre (landuse)

Segunda parte de la comparación del reto 1. El reto 1 trabajó con 3 escenas y 9
modelos; aquí se cambia el dataset (cobertura terrestre: `denseresidential`,
`harbor` y `river`) y se comparan **cuatro modelos** elegidos a partir de lo que
el reto 1 ya había demostrado. El objetivo no es repetir nueve entrenamientos,
sino comprobar si las conclusiones del reto 1 se mantienen en otro problema.

| Grupo | Modelo | Cuaderno | Resultados |
|---|---|---|---|
| CNN | CNN base (8-16-32, `Rescaling(1/255)` interno) | `LandUseCNN.ipynb` | `resultados/cnn/base/` |
| CNN | CNN mejorada con la receta v2 del reto 1 | `LandUseCNN.ipynb` | `resultados/cnn/mejorada/` |
| DL no secuencial | DenseNet121 preentrenada en ImageNet + fine-tuning | `LandUseDenseNet.ipynb` | `resultados/densenet/` |
| IA generativa | ViT-B/16 preentrenado en ImageNet + fine-tuning | `LandUseViT.ipynb` | `resultados/vit/` |

## Continuidad con el reto 1

El reto 1 comparó 9 modelos sobre 958 imágenes de 3 escenas. Estos fueron sus
resultados finales:

| Modelo del reto 1 | Accuracy | McNemar relevante |
|---|---:|---|
| CNN base | 90,42 % | vs DenseNet121: p = 0,0001 |
| CNN mejorada v1 | 87,08 % | vs CNN base: p = 0,2005 |
| CNN mejorada v2 | 89,17 % | vs CNN base: p = 0,7011; vs v1: p = 0,2668 |
| DenseNet121 | 98,33 % | — |
| ViT desde cero | 88,75 % | vs CNN v2: p = 1,0000; vs DenseNet121: p = 0,0000 |
| ViT preentrenado | 97,50 % | vs ViT desde cero: p = 0,0001; vs DenseNet121: p = 0,6875 |
| Mejor ML (SVM) | 95,42 % | vs DenseNet121: p = 0,0654 |

**Por qué se usan cuatro modelos.** Las tres CNN desde cero son
estadísticamente indistinguibles entre sí (p = 0,2005, 0,2668 y 0,7011), así que
conservar v1 y v2 no añade información. El ViT desde cero empata con la mejor
CNN desde cero (p = 1,0000): sin preentrenamiento, atención y convolución no se
separan. El mejor modelo clásico (SVM, 95,42 %) queda por debajo de DenseNet121
y no aporta una comparación nueva. Por eso el reto 2 conserva un representante
por celda útil: CNN base, CNN mejorada, DenseNet121 y ViT preentrenado.

**Por qué se parte del ViT preentrenado.** En el reto 1 el ViT desde cero fue el
modelo más caro y más difícil: 88,75 %, 2.345 s y 51 épocas, sin superar a la CNN
mejorada v2 (p = 1,0000). El mismo ViT con preentrenamiento subió al 97,50 %
(+8,75 puntos, p = 0,0001) y empató con DenseNet121 (p = 0,6875). Repetir el ViT
desde cero en el reto 2 solo confirmaría lo ya sabido y añadiría coste. El reto 2
arranca del preentrenado y se concentra en la pregunta que sigue abierta: si el
empate atención/convolución se mantiene en un dataset distinto.

## Dataset

- **1.500 imágenes PNG** del dataset local `landuse-dataset/`, 500 por clase:
  `denseresidential`, `harbor` y `river`. El dataset está equilibrado.
- Las imágenes originales son de 256×256. Las CNN y DenseNet trabajan a
  **128×128**; el ViT usa **224×224**, la resolución de su preentrenamiento.
- Split estratificado común: **75 % train, 15 % validación y 25 % test**,
  semilla 20 (`resultados/split.json`). El test tiene 375 imágenes y no se usa
  para entrenar ni para early stopping.

## Protocolo

- Métricas: accuracy, precisión/recall/F1 macro y weighted, Cohen's kappa,
  ROC-AUC one-vs-rest, matriz de confusión, IC 95 % por bootstrap (2.000
  remuestreos) y test exacto de McNemar.
- Cada modelo guarda `metrics.json`, `predictions.csv`,
  `classification_report.txt`, `history.csv` y sus pesos. El ViT guarda
  `model.weights.h5` (~345 MB), ignorado por git; en un clon nuevo se reentrena.
- Los modelos ya entrenados se cargan desde `resultados/` por defecto. Para
  reentrenar, cambia `FORCE_RETRAIN = True` en el cuaderno del modelo.
- `reto-2.ipynb` genera exactamente **cuatro figuras**: barras con IC 95 %,
  matrices de confusión, curvas de aprendizaje y heatmap de F1 por clase.

## Resumen de resultados

| Modelo | Familia | Accuracy | F1 macro | κ | ROC-AUC | Parámetros | Entreno (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN base | CNN | 89,33 % | 89,24 % | 0,840 | 0,9857 | 30.611 | 110,64 |
| CNN mejorada (v2) | CNN | 99,20 % | 99,20 % | 0,988 | 0,9993 | 305.251 | 8.338,90 |
| DenseNet121 | DL no secuencial | 98,93 % | 98,93 % | 0,984 | 0,9996 | 7.040.579 | 697,87 |
| ViT-B/16 | IA generativa (Transformer) | **99,47 %** | **99,47 %** | 0,992 | 1,0000 | 85.800.963 | 6.115,07 |

El **ViT-B/16 preentrenado** obtiene la mejor accuracy (99,47 %), seguido de la
CNN mejorada (99,20 %) y DenseNet121 (98,93 %). El test exacto de McNemar indica
que las diferencias entre estos tres modelos están dentro del ruido
(ViT frente a DenseNet121 p = 0,6875; ViT frente a la CNN mejorada p = 1,0000;
DenseNet121 frente a la CNN mejorada p = 1,0000). La CNN base sí queda
significativamente por debajo (p = 0,0000 frente a los demás). El detalle por
clase y los intervalos de confianza por bootstrap están en
`resultados/comparacion/`.

El análisis completo, las cuatro figuras y las conclusiones generadas desde
`resultados/` están en `reto-2.ipynb` y en `resultados/comparacion/`.

## Estructura de `resultados/`

```text
resultados/
├── split.json
├── cnn/base/
├── cnn/mejorada/
├── densenet/            # incluye gradcam.png
├── vit/                 # model.weights.h5 ignorado por git
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
`tensorflow`/`keras`, `keras-hub`, `scikit-learn`, `pandas`, `matplotlib`,
`seaborn`, `jupyter`/`nbconvert`, `reportlab` y `Pillow`. Están instalados en
el entorno `venv` del repositorio.

`keras-hub` se instala sin dependencias para no cambiar la versión de Keras del
entorno (en Python 3.14 no existe `tensorflow-text`, que el `ViTImageConverter`
necesita):

```bash
../../venv/bin/pip install --no-deps keras-hub kagglehub kagglesdk regex tqdm
```

## Reproducción

Desde esta carpeta y con el entorno virtual del repositorio:

```bash
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace LandUseCNN.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace LandUseDenseNet.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace LandUseViT.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace reto-2.ipynb
../../venv/bin/python informe_reto_2.py
```

`common_landuse.py` centraliza la carga de imágenes, el split, las métricas, el
guardado de resultados y la lógica de **cargar o reentrenar** para que todos los
cuadernos usen exactamente el mismo protocolo.

El ViT descarga el preset de ~328 MB la primera vez y guarda los pesos ajustados
en `resultados/vit/model.weights.h5` (~345 MB). Ese archivo y
`informe-reto-2.pdf` están en `.gitignore`; el repositorio guarda métricas,
predicciones e historia, y en un clon nuevo el cuaderno reentrena el modelo.

## Informe

`informe_reto_2.py` genera `informe-reto-2.pdf` a partir de `resultados/`:
portada, resumen ejecutivo, continuidad con el reto 1, tabla de resultados,
McNemar, las cuatro figuras y conclusiones. El PDF está ignorado por git; se
regenera con:

```bash
../../venv/bin/python informe_reto_2.py
```

## Diferencias deliberadas con el reto 1

- **Cuatro modelos frente a nueve.** Es una consecuencia de los resultados del
  reto 1, no una reducción por comodidad: las variantes eliminadas no cambiaban
  la conclusión ni el orden de los modelos.
- **Dataset distinto.** 1.500 imágenes y 375 de test frente a 958 y 240 del
  reto 1. El protocolo es el mismo, pero los IC son más estrechos y los
  p-valores no son directamente comparables entre retos.
- **ViT igual al del reto 1.** Mismo preset de KerasHub
  (`vit_base_patch16_224_imagenet`), mismos dos bloques descongelados y mismo
  esquema de dos fases. El reto 2 carga las imágenes directamente a 224×224
  desde los PNG de 256×256; el reto 1 las cargaba a 128 y las redimensionaba
  dentro del modelo. Es la única diferencia de preprocesado del ViT.
- **CNN mejorada alineada.** Ahora usa la receta v2 del reto 1
  (augmentación suave, `lr=3e-4`, `label_smoothing` y early stopping sobre
  `val_loss`) y `Rescaling(1/255)` dentro del modelo. Antes dividía las
  imágenes fuera y aplicaba la receta v1.
- **Sin ViT desde cero ni modelos clásicos.** El reto 1 ya demostró que no
  aportan una comparación nueva; repetirlos solo añadiría coste.
- **Pesos grandes fuera de git.** `resultados/vit/model.weights.h5` (~345 MB)
  está ignorado. El antiguo `resultados/vit/model.pt` de HuggingFace se eliminó
  al pasar al ViT de KerasHub.
- **Informe reproducible.** `informe_reto_2.py` genera `informe-reto-2.pdf`
  leyendo solo `resultados/`; ninguna cifra del PDF está escrita a mano.
