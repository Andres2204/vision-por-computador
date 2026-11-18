# Reto 1 — Clasificación de 3 escenas con aprendizaje profundo

Se parte del dataset de 3 escenas usado en la práctica de aprendizaje
automático (guía 2, reto 5) para clasificar **bosque** (`forest`), **costa**
(`coast`) y **autopista** (`highway`), y se comparan sobre el mismo split y con
las mismas métricas:

| Grupo | Modelo | Cuaderno | Resultados |
|---|---|---|---|
| CNN | CNN base (réplica del cuaderno del profesor, 8-16-32) | `ScenesCNN.ipynb` | `resultados/cnn/base/` |
| CNN | CNN mejorada (32-64-128, dos convoluciones por bloque, BN + Dropout + GAP + Dense(128) + augmentación) | `ScenesCNN.ipynb` | `resultados/cnn/mejorada/` |
| CNN | CNN mejorada v2 (segunda iteración: augmentación suave, lr 3e-4, `label_smoothing` y early stopping sobre `val_loss`) | `ScenesCNN.ipynb` | `resultados/cnn/mejorada_v2/` |
| DL no secuencial | DenseNet121 preentrenada en ImageNet + fine-tuning en dos fases | `ScenesDenseNet.ipynb` | `resultados/densenet/` (+ `gradcam.png`) |
| IA generativa | ViT (Vision Transformer) entrenado desde cero (atención global sobre todos los parches) | `ScenesViT.ipynb` | `resultados/vit/` |
| Machine learning | Regresión logística (HSV+HOG, escalada) | `ScenesClassicML.ipynb` | `resultados/ml/logreg/` |
| Machine learning | SVM (kernel RBF, C=10) | `ScenesClassicML.ipynb` | `resultados/ml/svm/` |
| Machine learning | Random Forest (300 árboles) | `ScenesClassicML.ipynb` | `resultados/ml/rf/` |
| Comparación | Tabla resumen, 4 figuras, McNemar y conclusiones | `reto-1.ipynb` | `resultados/comparacion/` |

El informe se genera a partir de `resultados/` y se guarda como
`informe-reto-1.pdf` en esta carpeta (ignorado por git): resumen, protocolo,
tabla de los modelos, McNemar, las 4 figuras y conclusiones.

La arquitectura **no secuencial** que pide el enunciado es `DenseNet121`
preentrenada en ImageNet con fine-tuning en dos fases (base congelada y
descongelación de las últimas 40 capas manteniendo las `BatchNormalization`
congeladas). La bala de **IA generativa** la cubre un **ViT** entrenado
**desde cero** (`ScenesViT.ipynb`; el enunciado admite «ViT, SWIN u otro»):
parches de 8×8 (16×16 = 256 tokens), atención **global** (todos los parches
atienden a todos, `O(n²)`) y una arquitectura **plana** de seis bloques
idénticos (dim 256, 8 cabezas, MLP 1024, dropout 0.1), sin etapas jerárquicas.
No usa pesos de ImageNet, de modo que la comparación con DenseNet121 separa
*arquitectura* (atención vs convolución) de *preentrenamiento* (ImageNet vs
desde cero), y la comparación con la CNN mejorada v2 —que también arranca
desde cero— aísla la arquitectura.

El ViT es el modelo **más difícil de entrenar** con este dataset: un
Transformer no trae los sesgos inductivos de las convoluciones (localidad,
compartición de pesos) y tiene que aprenderlos de los datos, pero solo hay
~610 imágenes de train. La augmentación, el `weight_decay` de AdamW y el early
stopping son imprescindibles y, aun así, cabe esperar que quede por detrás de
la DenseNet121 preentrenada: el hueco entre ambos mezcla arquitectura y
preentrenamiento, y `reto-1.ipynb` lo descompone con el McNemar y los IC de
bootstrap. Además, sus capas personalizadas no se deserializan con fiabilidad,
así que se guarda también `model.weights.h5` y se recarga reconstruyendo la
arquitectura con `build_vit()`.

## Dataset

* 958 imágenes JPG utilizables: `coast/` (360), `forest/` (328) y
  `highway/` (270), es decir, un reparto aproximado de 38/34/28 % (dataset
  desbalanceado). Cada carpeta incluye además un `desktop.ini` que no se
  carga.
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
  (semilla 20 → ≈ 610 train / 108 validación / 240 test; se guarda en
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
│   ├── mejorada/
│   └── mejorada_v2/
├── densenet/            # incluye gradcam.png
├── vit/                 # incluye model.weights.h5 + ejemplos_prediccion.png
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
| CNN base | 90.42 % | 90.12 % | 30 611 | 6.5 s |
| CNN mejorada | 87.08 % | 86.08 % | 305 251 | 482 s |
| CNN mejorada v2 | 89.17 % | 88.50 % | 305 251 | 963 s |
| DenseNet121 | 98.33 % | 98.36 % | 7 040 579 (818 179 entrenables) | 128 s |
| ViT (desde cero) | pendiente de entrenar | pendiente de entrenar | n/d | n/d |
| Regresión logística | 94.17 % | 93.91 % | n/d | 0.46 s |
| SVM (RBF) | 95.42 % | 95.26 % | n/d | 0.37 s |
| Random Forest | 93.33 % | 93.00 % | n/d | 1.59 s |

Resultados obtenidos el 17/11/2026 ejecutando los cuadernos con el
entorno `.venv` del repositorio (240 imágenes de test, las mismas para todos).
DenseNet121 gana con un IC 95 % de 96.67–99.58 %. McNemar: DenseNet121 vs CNN
base p = 0.0001, vs CNN mejorada p < 0.0001 y vs CNN mejorada v2 p < 0.0001
(diferencias significativas); CNN base vs CNN mejorada p = 0.2005, CNN mejorada
v2 vs base p = 0.7011, v2 vs v1 p = 0.2668 y SVM vs DenseNet121 p = 0.0654. La
v2 sube a 89.17 %: +2.1 puntos sobre la v1 y −1.2 respecto a la base (ninguna
de las dos diferencias es significativa). La tabla completa está en
`resultados/comparacion/model_summary.csv`; el ViT entra en ella y en el
McNemar en cuanto se ejecute `ScenesViT.ipynb`.

La CNN mejorada se entrenó **dos veces**, y eso es parte del resultado. La
primera versión (v1, 87.08 %) quedó 3.3 puntos por debajo de la base: su
`history.csv` muestra el `val_accuracy` clavado en 0.3796 durante las primeras
13 épocas con el `val_loss` subiendo hasta 2.8, y el entrenamiento se cortó en
la época 40 cuando la validación todavía venía mejorando. Diagnóstico:
augmentación demasiado agresiva para 610 imágenes, `lr = 1e-3` con
`BatchNormalization` y `EarlyStopping` sobre una `val_accuracy` ruidosa (108
imágenes). La **v2** (`resultados/cnn/mejorada_v2/`) cambia solo el régimen de
entrenamiento —augmentación a flip + rotación 0.05, `lr = 3e-4`,
`label_smoothing = 0.05` en la función de costo y paciencia 20 sobre
`val_loss`— y deja intacta
la arquitectura: es la evidencia experimental del bullet «ajustar los
hiperparámetros o la función de costo» del enunciado. El resultado es 89.17 %
(+2.1 puntos sobre la v1), todavía 1.2 puntos por debajo de la base; McNemar
no separa ninguna de las dos diferencias (p = 0.2668 y p = 0.7011). La v1 se
conserva como evidencia de la iteración.

### Comparación con el machine learning original de la guía 2

Las cifras se transcriben de las salidas de `guia-2/reto-5/reto-5.ipynb` (no
se reentrenan). Ese cuaderno usa el mismo protocolo (25 % de test, semilla 20,
estratificado, 240 imágenes), pero carga las imágenes en otro orden, así que la
comparación es orientativa y no entra en el McNemar.

| Modelo (guía 2) | Características | Accuracy CV | Accuracy test |
|---|---|---|---|
| Random Forest | RGB + HSV + Haralick | 90.11 % | 90.00 % |
| MLP | RGB + HSV + Haralick | 90.67 % | 89.17 % |
| SVC (RBF), elegido por CV | RGB + HSV + Haralick | 90.81 % | 88.75 % |
| MLP (línea base) | medias/desv. RGB | — | 81.25 % |

* Frente al ML del inicio del curso, el deep learning sí mejora: DenseNet121
  (98.33 %) le saca 8–10 puntos y la CNN base del profesor (90.42 %) ya lo
  iguala sin diseñar características a mano.
* Las características pesan tanto como el modelo: el mismo SVM pasa de
  88.75 % (Haralick) a 95.42 % (HOG), y con HOG el ML clásico supera a todas
  las redes entrenadas desde cero.
* Con ~610 imágenes de train, lo que más aporta es el preentrenamiento: las
  redes entrenadas desde cero se quedan en la banda del mejor modelo de la
  guía 2 (90 %) o por debajo, mientras la DenseNet121 llega a 98.33 %.

El detalle (los 9 modelos de la guía 2 y la tabla conjunta) está en la sección
«Comparación con el machine learning original de la guía 2» de `reto-1.ipynb`.

## Notas técnicas

* **`Adam(decay=...)` y Keras 3.** El cuaderno original del profesor usaba
  `Adam(learning_rate=1e-3, decay=1e-3/50)`. El argumento `decay` fue
  eliminado en Keras 3 (los schedules son objetos o callbacks), así que la CNN
  base replica se compila con `Adam(1e-3)` sin decay; la CNN mejorada gestiona
  el learning rate con `ReduceLROnPlateau`.
* **Pesos de ImageNet.** `DenseNet121(weights="imagenet")` descarga los pesos
  en la primera ejecución de `ScenesDenseNet.ipynb`; es el único acceso a
  internet de todo el reto.
* **Augmentación dentro del modelo.** Las capas `Random*` viven dentro de las
  arquitecturas y Keras las desactiva automáticamente en inferencia.
* **Early stopping de la CNN mejorada.** La v1 monitoriza `val_accuracy`
  (paciencia 8): con 108 imágenes de validación el mínimo de `val_loss` puede
  caer en la primera época y `restore_best_weights` devolvería un modelo sin
  entrenar. La v2, con `lr` más bajo y augmentación suave, vuelve a
  `val_loss` con paciencia 20.
* **`class_weight`.** La CNN mejorada pondera las clases para compensar el
  desbalance 38/34/28 % del train.
* **Grad-CAM.** `ScenesDenseNet.ipynb` guarda en
  `resultados/densenet/gradcam.png` una imagen de test con su mapa de calor
  para cada una de las tres clases; la celda está protegida con `try/except`
  para que el cuaderno nunca se rompa por la figura.
* **Recarga del ViT.** Las capas personalizadas del ViT (`PatchEmbedding`,
  `PatchEncoder`, `TransformerBlock`) no se deserializan con fiabilidad desde
  `model.keras`, así que `ScenesViT.ipynb` guarda además `model.weights.h5` y
  `load_vit` reconstruye la arquitectura con `build_vit()` antes de cargar los
  pesos.

## Dependencias

Además de `numpy`, `opencv-python` y `scipy`, estos cuadernos usan
`tensorflow`/`keras` (Keras 3), `scikit-learn`, `scikit-image` (HOG),
`pandas`, `matplotlib`, `seaborn` y `joblib` (incluido con scikit-learn).
Están instalados en el entorno `venv` del repositorio.

## Reproducción

Desde esta carpeta y con el entorno virtual del repositorio (primera
ejecución: entrena todos los modelos (incluida la variante v2 de la CNN);
siguientes ejecuciones: cargan los artefactos guardados):

```bash
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesCNN.ipynb
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesDenseNet.ipynb
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesClassicML.ipynb
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace ScenesViT.ipynb
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace reto-1.ipynb
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
   completa de p-valores; el SPEC pide comparaciones concretas (base vs
   mejorada, mejorada vs DenseNet, base vs DenseNet y, desde la segunda
   entrega, el ViT contra la CNN mejorada v2 y contra DenseNet), más el mejor
   ML vs el mejor DL, así que el CSV guarda una fila por comparación en
   formato largo.
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
7. **Entorno virtual.** El repositorio usa `.venv/` en la raíz (ignorado por
   git), no `venv/`; los comandos de reproducción apuntan a
   `../../.venv/bin/jupyter`.
8. **Hiperparámetros del ViT en `metrics.json`.** El SPEC de esta entrega
   exige registrar la configuración del Transformer; el ViT guarda `input`,
   `patch_size`, `embed_dim`, `depth`, `num_heads`, `mlp_dim`, `dropout`,
   `label_smoothing`, `optimizer`, `lr`, `weight_decay`, `batch_size`,
   `epochs` y `trained_from_scratch: True`.
