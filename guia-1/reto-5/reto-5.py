import cv2
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)


# 1. LEER IMAGEN LIMPIA (se redimensiona un poco para que los filtros
#    manuales, con ventana deslizante en numpy puro, corran rápido)
ruta = "guia-1/reto-5/images.jpg"
img_original = cv2.imread(ruta, cv2.IMREAD_COLOR)
img_limpia = img_original.astype(np.float64)
print("Shape imagen de trabajo:", img_limpia.shape)



# 2. AÑADIR RUIDO SALT & PEPPER (implementación propia, sin OpenCV)
def salt_pepper_noise(img, prob=0.05):
    """
    Añade ruido sal y pimienta a una imagen (color o gris).
    prob: probabilidad total de que un píxel se vea afectado
          (mitad se pone en 255 -> 'sal', mitad en 0 -> 'pimienta')
    """
    ruidosa = img.copy()
    h, w = img.shape[:2]
    mascara = rng.random((h, w))

    # Sal (blanco) y pimienta (negro) se aplican a TODOS los canales
    # del píxel para que se vea como un punto blanco/negro real.
    ruidosa[mascara < prob / 2] = 0                            # pimienta (negro)
    ruidosa[(mascara >= prob / 2) & (mascara < prob)] = 255    # sal (blanco)
    return ruidosa


img_ruidosa = salt_pepper_noise(img_limpia, prob=0.06)



# 3. CONVOLUCIÓN 2D IMPLEMENTADA DESDE CERO (sin cv2.filter2D)
#    Usamos sliding_window_view de numpy solo para vectorizar el
#    recorrido de ventanas; el cómputo (suma ponderada) es manual.
def pad_imagen(canal, pad, modo="reflect"):
    return np.pad(canal, ((pad, pad), (pad, pad)), mode=modo)


def convolucion_manual(canal, kernel):
    """Convolución 2D 'a mano' de un canal (2D) con un kernel 2D."""
    k = kernel.shape[0]
    pad = k // 2
    canal_pad = pad_imagen(canal, pad)
    ventanas = np.lib.stride_tricks.sliding_window_view(canal_pad, (k, k))
    # Producto elemento a elemento y suma -> definición de convolución/correlación
    salida = np.tensordot(ventanas, kernel, axes=([2, 3], [0, 1]))
    return salida


def aplicar_a_color(img, funcion_canal, *args, **kwargs):
    """Aplica una función definida para un canal 2D a cada canal B,G,R."""
    canales = [funcion_canal(img[:, :, c], *args, **kwargs) for c in range(3)]
    return np.stack(canales, axis=2)


# --- Kernel de media (promedio simple) ---
def kernel_media(k=3):
    return np.ones((k, k), dtype=np.float64) / (k * k)


# --- Kernel Gaussiano (calculado a mano, sin cv2.getGaussianKernel) ---
def kernel_gaussiano(k=5, sigma=1.0):
    ax = np.arange(-(k // 2), k // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()



# 3b. FILTRO DE MEDIANA IMPLEMENTADO A MANO (no es convolución lineal,
#     pero se pide explícitamente como alternativa para quitar ruido)
def filtro_mediana_manual(canal, k=3):
    pad = k // 2
    canal_pad = pad_imagen(canal, pad)
    ventanas = np.lib.stride_tricks.sliding_window_view(canal_pad, (k, k))
    return np.median(ventanas, axis=(2, 3))



# 4. APLICAR LOS FILTROS MANUALES SOBRE LA IMAGEN RUIDOSA
img_media = aplicar_a_color(img_ruidosa, convolucion_manual, kernel_media(3))
img_gauss = aplicar_a_color(img_ruidosa, convolucion_manual, kernel_gaussiano(5, 1.0))
img_mediana_manual = aplicar_a_color(img_ruidosa, filtro_mediana_manual, 3)


# 5. FUNCIONES PROPIAS DE OPENCV, UNA POR CADA FILTRO MANUAL
img_ruidosa_u8 = img_ruidosa.astype(np.uint8)

# --- Equivalente a la MEDIA manual -> cv2.blur (o cv2.boxFilter) ---
img_media_cv2 = cv2.blur(img_ruidosa_u8, (3, 3)).astype(np.float64)

# --- Equivalente al GAUSSIANO manual -> cv2.GaussianBlur ---
img_gauss_cv2 = cv2.GaussianBlur(img_ruidosa_u8, (5, 5), sigmaX=1.0).astype(np.float64)

# --- Equivalente a la MEDIANA manual -> cv2.medianBlur ---
img_mediana_cv2 = cv2.medianBlur(img_ruidosa_u8, 3).astype(np.float64)




# 6. MÉTRICAS DE COMPARACIÓN: MSE Y PSNR RESPECTO A LA IMAGEN LIMPIA
def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)


def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / m)


resultados = {
    "Ruidosa (sin filtrar)": img_ruidosa,
    "Media 3x3 (manual)": img_media,
    "Media 3x3 (cv2.blur)": img_media_cv2,
    "Gaussiano 5x5 (manual)": img_gauss,
    "Gaussiano 5x5 (cv2.GaussianBlur)": img_gauss_cv2,
    "Mediana 3x3 (manual)": img_mediana_manual,
    "Mediana 3x3 (cv2.medianBlur)": img_mediana_cv2,
}

print("\n{:<32}{:>12}{:>12}".format("Método", "MSE", "PSNR (dB)"))
print("-" * 56)
for nombre, imagen in resultados.items():
    m = mse(imagen, img_limpia)
    p = psnr(imagen, img_limpia)
    print("{:<32}{:>12.2f}{:>12.2f}".format(nombre, m, p))


# 7. GUARDAR IMÁGENES PARA VISUALIZACIÓN (collage comparativo)
def a_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


etiquetas = ["Limpia", "Ruidosa",
             "Media (manual)", "Media (cv2.blur)",
             "Gaussiano (manual)", "Gaussiano (cv2.GaussianBlur)",
             "Mediana (manual)", "Mediana (cv2.medianBlur)"]
imagenes = [img_limpia, img_ruidosa,
            img_media, img_media_cv2,
            img_gauss, img_gauss_cv2,
            img_mediana_manual, img_mediana_cv2]

h, w = img_limpia.shape[:2]
cols = 3
rows = 3
collage = np.ones((rows * (h + 40), cols * (w + 10), 3), dtype=np.uint8) * 255

for i, (etq, im) in enumerate(zip(etiquetas, imagenes)):
    r, c = divmod(i, cols)
    y0 = r * (h + 40) + 30
    x0 = c * (w + 10) + 5
    collage[y0:y0 + h, x0:x0 + w] = a_uint8(im)
    cv2.putText(collage, etq, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite("guia-1/reto-5/collage_filtros.png", collage)
cv2.imwrite("guia-1/reto-5/img_limpia.png", a_uint8(img_limpia))
cv2.imwrite("guia-1/reto-5/img_ruidosa.png", a_uint8(img_ruidosa))
cv2.imwrite("guia-1/reto-5/img_media_manual.png", a_uint8(img_media))
cv2.imwrite("guia-1/reto-5/img_media_cv2.png", a_uint8(img_media_cv2))
cv2.imwrite("guia-1/reto-5/img_gauss_manual.png", a_uint8(img_gauss))
cv2.imwrite("guia-1/reto-5/img_gauss_cv2.png", a_uint8(img_gauss_cv2))
cv2.imwrite("guia-1/reto-5/img_mediana_manual.png", a_uint8(img_mediana_manual))
cv2.imwrite("guia-1/reto-5/img_mediana_cv2.png", a_uint8(img_mediana_cv2))
print("\nImágenes guardadas correctamente.")