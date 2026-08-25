# Curso de Vision por Computador

Repositorio de prácticas de la asignatura. Los retos están en Python y algunos
usan OpenCV para procesamiento de imágenes y captura de vídeo.

---

## Requisitos

- Python 3.12 o superior (probado con Python 3.14)
- Entorno virtual `venv` (recomendado) o instalación aislada de las
  dependencias listadas en `requirements.txt`.

### Crear el entorno virtual e instalar dependencias

```bash
cd vision-por-computador
python3 -m venv venv
source venv/bin/activate          # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Si el gestor de paquetes del sistema bloquea `ensurepip` por ser un entorno
> "externally managed" (Arch Linux / PEP 668), no importa: `python3 -m venv`
> incluye su propio `pip` embebido y funciona sin necesidad de `ensurepip`.

---

## Estructura

```
vision-por-computador/
├── README.md
├── requirements.txt
└── guia-1/
    ├── reto-1/   # matriz aleatoria + estadística (sin dependencias externas)
    ├── reto-3/   # ROI y estadística de color sobre una imagen
    ├── reto-4/   # captura de webcam en vivo + segmentación HSV
    └── reto-5/   # ruido sal-pimienta, filtros y métricas MSE/PSNR
```

> **Importante:** los retos 3 y 5 usan rutas relativas al **raíz del proyecto**
> (p. ej. `"guia-1/reto-3/keukenhof-paises-bajos.webp"`). Por eso **todos los
> retos deben ejecutarse desde la carpeta `vision-por-computador/`**, no desde
> dentro de la carpeta de cada reto.

---

## Cómo ejecutar cada reto

Activa el entorno virtual antes de lanzar cualquier reto:

```bash
source venv/bin/activate
```

### reto-1 — Estadística de una matriz aleatoria

```bash
python guia-1/reto-1/reto-1.py
```

- **Qué hace:** genera una matriz 1000×1000 de valores aleatorios entre 0 y
  255, calcula dimensiones, mínimo, máximo, media y desviación estándar, y
  vuelca la matriz en `guia-1/reto-1/output.txt`.
- **Dependencias:** solo la librería estándar (`random`, `math`). No necesita
  `cv2`/`numpy`/`scipy`, aunque el script incluye un bloque OpenCV comentado
  para visualizar la matriz como imagen.

### reto-3 — ROI y estadística de color

```bash
python guia-1/reto-3/reto-3.py
```

- **Qué hace:** lee la imagen `guia-1/reto-3/keukenhof-paises-bajos.webp`,
  dibuja 6 regiones de interés (ROI) sobre una copia y la guarda como
  `guia-1/reto-3/keukenhof_regiones.png`. Luego imprime, por cada ROI, la
  media y la desviación estándar de los canales B, G y R.
- **Dependencias:** `cv2`, `numpy` (vía `requirements.txt`).

### reto-5 — Ruido, filtros y métricas de calidad

```bash
python guia-1/reto-5/reto-5.py
```

- **Qué hace:** parte de `guia-1/reto-5/images.jpg`, le añade ruido
  sal-pimienta y aplica tres filtros (media, gaussiano, mediana) cada uno
  implementado a mano y con su equivalente de OpenCV. Compara los
  resultados con MSE y PSNR respecto a la imagen limpia y guarda un collage
  comparativo en `guia-1/reto-5/collage_filtros.png` más las imágenes
  individuales.
- **Dependencias:** `cv2`, `numpy`, `scipy` (vía `requirements.txt`).

### reto-4 — Webcam en vivo + segmentación HSV

```bash
QT_QPA_PLATFORM=xcb python guia-1/reto-4/reto-4.py
```

- **Qué hace:** abre la cámara web (`VideoCapture(0)`), muestra el vídeo en
  vivo en una ventana y, en otra, la máscara binaria de los píxeles azules
  según segmentación HSV, dibujando los contornos más grandes.
- **Controles:** pulsa `ESC` o `q` para salir, o cierra la ventana "camera".
- **Requisitos:** cámara web conectada (`/dev/video0`) y sesión gráfica.
- **Dependencias:** `cv2`, `numpy` (vía `requirements.txt`).
- **Nota para Wayland:** `opencv-python` no incluye el plugin de Qt para
  Wayland, por lo que en sesiones Wayland hay que forzar el backend X11
  con `QT_QPA_PLATFORM=xcb` (requiere XWayland, habitual en la mayoría de
  escritorios modernos).

---

## Notas

- Los scripts se ejecutan desde la raíz del proyecto
  (`vision-por-computador/`) por las rutas relativas que usan.
- El directorio `venv/` está ignorado por git (ver `.gitignore`).
- En Linux con entorno "externally managed" (PEP 668) no se puede instalar
  paquetes a nivel de sistema con `pip`; por eso el uso de `venv` es el
  procedimiento recomendado.