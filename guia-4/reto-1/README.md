# Reto 1 — Clasificación de 3 escenas con aprendizaje profundo

Se parte del dataset de 3 escenas usado en la práctica de aprendizaje
automático (guía 2, reto 5) para clasificar **bosque** (`forest`), **costa**
(`coast`) y **autopista** (`highway`), y se comparan sobre el mismo split y con
las mismas métricas:

| Grupo | Modelo | Cuaderno | Resultados |
|---|---|---|---|
| CNN | CNN base (réplica del cuaderno del profesor, 8-16-32) | `ScenesCNN.ipynb` | `resultados/cnn/base/` |
| CNN | CNN mejorada (32-64-128 + BN + Dropout + GAP + augmentación) | `ScenesCNN.ipynb` | `resultados/cnn/mejorada/` |
| DL no secuencial | DenseNet121 preentrenada en ImageNet + fine-tuning en dos fases | `ScenesDenseNet.ipynb` | `resultados/densenet/` (+ `gradcam.png`) |
| Machine learning | Regresión logística (HSV+HOG, escalada) | `ScenesClassicML.ipynb` | `resultados/ml/logreg/` |
| Machine learning | SVM (kernel RBF, C=10) | `ScenesClassicML.ipynb` | `resultados/ml/svm/` |
| Machine learning | Random Forest (300 árboles) | `ScenesClassicML.ipynb` | `resultados/ml/rf/` |
| Comparación | Tabla resumen, 4 figuras, McNemar y conclusiones | `reto-1.ipynb` | `resultados/comparacion/` |

La arquitectura **no secuencial** que pide el enunciado es `DenseNet121`
preentrenada en ImageNet con fine-tuning en dos fases (base congelada y
descongelación de las últimas 40 capas manteniendo las `BatchNormalization`
congeladas). La comparación con **IA generativa** (Transformer tipo ViT/SWIN)
llega en la entrega siguiente: `resultados/` está preparado para que solo haya
que añadir una carpeta más con el mismo esquema.

## Dataset

* 961 imágenes JPG: `coast/` (361), `forest/` (329) y `highway/` (271), es
  decir, un reparto aproximado de 38/34/28 % (dataset desbalanceado).
* Ubicación esperada (en este orden; `common_3scenes.py` usa la primera que
  exista): `3scenes/` junto a este módulo, o
  `../../guia-2/reto-5/3scenes/` (la ubicación real del repo).
* El dataset está ignorado por git (`/**/3scenes/`): **no** se copia ni se
  sube al repositorio.
* Las imágenes se cargan con OpenCV (BGR→RGB), se redimensionan a 128×128 con
  `INTER_AREA` y se devuelven `uint8` sin normalizar: la normalización vive
  dentro de cada modelo (`Rescaling(1/255)` en las CNN, `preprocess_input` de
  DenseNet en la Lambda del modelo).

## Protocolo

* Split común: 25 % test y, del 75 % restante, un 15 % para validación
  (semilla 20 → ≈ 612 train / 108 validación / 241 test; se guarda en
  `resultados/split.json`). Todos los modelos
  evalúan exactamente el mismo `test_idx` (~240 imágenes); ningún cuaderno
  vuelve a hacer `train_test_split`.
* Métricas: accuracy, precisión/recall/F1 macro y weighted, Cohen's kappa,
  ROC-AUC one-vs-rest, matriz de confusión, IC 95 % por bootstrap (2000
  remuestreos) y test exacto de McNemar. El esquema de `metrics.json` es
  idéntico al del reto 2.
* Todos los modelos se cargan desde `resultados/` por defecto. Para
  reentrenar, cambia `FORCE_RETRAIN = True` en la celda de configuración del
  cuaderno correspondiente.
* Cada carpeta de modelo guarda `metrics.json`, `predictions.csv`,
  `classification_report.txt`, `history.csv` (solo modelos con épocas) y el
  modelo entrenado (`model.keras`; en `resultados/ml/` es `model.joblib`,
  porque los estimadores de scikit-learn no usan el formato `.keras`).
* `reto-1.ipynb` genera las **4 figuras** de la comparación (barras de
  accuracy con IC 95 %, matrices de confusión, curvas de aprendizaje y heatmap
  de F1 por clase) y las conclusiones **se generan desde los artefactos**
  (ninguna cifra está escrita a mano).

## Estructura de `resultados/`

```text
resultados/
├── split.json
├── cnn/
│   ├── base/
│   └── mejorada/
├── densenet/            # incluye gradcam.png
├── ml/
│   ├── logreg/          # model.joblib, sin history.csv
│   ├── svm/
│   └── rf/
└── comparacion/
    ├── model_summary.csv
    ├── mcnemar_pvalues.csv
    ├── barras_accuracy_ic95.png
    ├── matrices_confusion.png
    ├── curvas_aprendizaje.png
    └── heatmap_f1_por_clase.png
```

## Resumen de resultados

| Modelo | Accuracy | F1 macro | Parámetros | Entrenamiento |
|---|---|---|---|---|
| CNN base | pendiente de entrenar | pendiente de entrenar | pendiente de entrenar | pendiente de entrenar |
| CNN mejorada | pendiente de entrenar | pendiente de entrenar | pendiente de entrenar | pendiente de entrenar |
| DenseNet121 | pendiente de entrenar | pendiente de entrenar | pendiente de entrenar | pendiente de entrenar |
| Regresión logística | pendiente de entrenar | pendiente de entrenar | n/d | pendiente de entrenar |
| SVM (RBF) | pendiente de entrenar | pendiente de entrenar | n/d | pendiente de entrenar |
| Random Forest | pendiente de entrenar | pendiente de entrenar | n/d | pendiente de entrenar |

Los cuadernos se han entregado **sin ejecutar** (sin entrenar, sin descargar
nada): al ejecutarlos por primera vez se rellenan `resultados/` y esta tabla
(`resultados/comparacion/model_summary.csv`). Mientras tanto, `reto-1.ipynb`
imprime un aviso y termina sin error.

## Notas técnicas

* **`Adam(decay=...)` y Keras 3.** El cuaderno original del profesor usaba
  `Adam(learning_rate=1e-3, decay=1e-3/50)`. El argumento `decay` fue
  eliminado en Keras 3 (los schedules son objetos o callbacks), así que la CNN
  base replica se compila con `Adam(1e-3)` sin decay; la CNN mejorada gestiona
  el learning rate con `ReduceLROnPlateau`.
* **Pesos de ImageNet.** `DenseNet121(weights="imagenet")` descarga los pesos
  en la primera ejecución de `ScenesDenseNet.ipynb`; es el único acceso a
  internet de todo el reto (aquí no se ha ejecutado, así que no se ha
  descargado nada).
* **Augmentación dentro del modelo.** Las capas `Random*` viven dentro de las
  arquitecturas y Keras las desactiva automáticamente en inferencia.
* **`class_weight`.** La CNN mejorada pondera las clases para compensar el
  desbalance 38/34/28 % del train.
* **Grad-CAM.** `ScenesDenseNet.ipynb` guarda en
  `resultados/densenet/gradcam.png` una imagen de test con su mapa de calor
  para cada una de las tres clases; la celda está protegida con `try/except`
  para que el cuaderno nunca se rompa por la figura.

## Dependencias

Además de `numpy`, `opencv-python` y `scipy`, estos cuadernos usan
`tensorflow`/`keras` (Keras 3), `scikit-learn`, `scikit-image` (HOG),
`pandas`, `matplotlib`, `seaborn` y `joblib` (incluido con scikit-learn).
Están instalados en el entorno `venv` del repositorio.

## Reproducción

Desde esta carpeta y con el entorno virtual del repositorio (primera
ejecución: entrena los seis modelos; siguientes ejecuciones: cargan los
artefactos guardados):

```bash
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesCNN.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesDenseNet.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesClassicML.ipynb
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace reto-1.ipynb
```

`common_3scenes.py` centraliza la carga de imágenes (con resolución de la ruta
del dataset), el split, las métricas, el guardado de resultados y la lógica de
**cargar o reentrenar** para que todos los cuadernos usen exactamente el mismo
protocolo.

## Dudas y decisiones bajo el SPEC

Siguiendo la indicación del SPEC (sec. 0), se documentan aquí las dudas:

1. **`barras_accuracy_ic95.png` vs `barras_accuracy_f1.png`.** El SPEC pide
   explícitamente barras de *accuracy con IC 95 %* con ese nombre de archivo,
   distinto del de reto 2 (`barras_accuracy_f1.png`, accuracy + F1). Se ha
   seguido el SPEC.
2. **`mcnemar_pvalues.csv` con formato largo.** Reto 2 guardaba una matriz
   completa de p-valores; el SPEC de este reto pide solo cuatro comparaciones
   (base vs mejorada, mejorada vs DenseNet, base vs DenseNet, mejor ML vs
   mejor DL), así que el CSV guarda esas cuatro filas en formato largo.
3. **`model.joblib` en `resultados/ml/`.** El SPEC exige reutilizar
   `load_or_train`, pero los estimadores de sklearn no guardan en `.keras`;
   se añade el artefacto `model.joblib` (y su carga con joblib) para que
   también cumplan la lógica de cargar-o-reentrenar. Los demás nombres de
   artefactos y las claves de `metrics.json` son idénticos a reto 2. Los
   modelos clásicos no tienen `history.csv` ni conteo de parámetros (se
   registran como `null`, como admite el esquema).
4. **`Rescaling` dentro de las dos CNN.** El SPEC exige que la CNN base
   replique el cuaderno del profesor (con `Rescaling(1/255)` dentro), pero no
   dice nada de la normalización de la CNN mejorada; se ha optado por incluir
   también `Rescaling(1/255)` tras la augmentación, de modo que los datos
   floten en bruto (0-255) por todos los cuadernos y cada modelo posea su
   normalización (igual que DenseNet posee `preprocess_input`).
5. **Mensajes impresos en español en los cuadernos.** La política de idioma
   del SPEC exige español para "cualquier mensaje impreso para el estudiante",
   mientras que los cuadernos de referencia imprimen en inglés; se ha seguido
   el SPEC (los `print` de `common_3scenes.py` replican los de
   `common_landuse.py`, que están en inglés).
6. **Modelos clásicos ajustados solo con train.** sklearn no usa early
   stopping, así que la validación no interviene en su ajuste; se entrenan con
   el train del split común (no con train+val) para mantener la comparabilidad
   con los modelos de deep learning, que también se ajustan solo con train.
7. **`../../venv` no existe en la máquina de edición.** Los comandos de
   reproducción usan la ruta que exige el SPEC; en este equipo el entorno no
   está creado, pero la ruta es la canónica del repositorio.
