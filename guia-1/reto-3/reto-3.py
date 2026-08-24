import cv2
import numpy as np

# LEER LA IMAGEN A COLOR CON OPENCV
ruta = "reto-3/keukenhof-paises-bajos.webp"
img_bgr = cv2.imread(ruta, cv2.IMREAD_COLOR)


# IMPRIMIR LA FORMA (SHAPE) DE LA IMAGEN
print("Shape de la imagen (alto, ancho, canales):", img_bgr.shape)


# DEFINIR LAS 6 REGIONES DE INTERÉS (ROI) -> (x, y, w, h)
# Elegidas por inspección visual
# ROI1-5: zonas de color predominantemente homogéneo
# ROI6  : zona heterogénea (casas + molinos + cielo + gente)
rois = {
    "ROI1_Rojo_tulipanes":      (170, 250, 50, 40),
    "ROI2_Amarillo_tulipanes":  (290, 250, 50, 40),
    "ROI3_Magenta_tulipanes":   (510, 240, 50, 40),
    "ROI4_Blanco_rosado":       (15, 300, 50, 30),
    "ROI5_Celeste_cielo":       (250, 15, 80, 40),
    "ROI6_Mixta_casas_molinos": (80, 150, 250, 50),
}

colores_dibujo = {
    "ROI1_Rojo_tulipanes":      (255, 255, 255),
    "ROI2_Amarillo_tulipanes":  (255, 0, 0),
    "ROI3_Magenta_tulipanes":   (255, 255, 255),
    "ROI4_Blanco_rosado":       (0, 0, 0),
    "ROI5_Celeste_cielo":       (0, 0, 255),
    "ROI6_Mixta_casas_molinos": (0, 0, 0),
}


# DIBUJAR LOS RECUADROS SOBRE UNA COPIA DE LA IMAGEN ORIGINAL
img_anotada = img_bgr.copy()
for nombre, (x, y, w, h) in rois.items():
    color = colores_dibujo[nombre]
    cv2.rectangle(img_anotada, (x, y), (x + w, y + h), color, 2)
    etiqueta = nombre.split("_")[0]
    ty = y - 6 if y - 6 > 10 else y + h + 15
    cv2.putText(img_anotada, etiqueta, (x, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

cv2.imwrite("reto-3/keukenhof_regiones.png", img_anotada)


# MEDIA Y DESVIACIÓN ESTÁNDAR POR CANAL (B, G, R) EN CADA ROI
print("\n{:<28}{:>20}{:>28}".format("Región", "Media (B,G,R)", "Desv. estándar (B,G,R)"))
print("-" * 76)

for nombre, (x, y, w, h) in rois.items():
    region = img_bgr[y:y + h, x:x + w]
    media, std = cv2.meanStdDev(region)
    media = media.flatten()
    std = std.flatten()
    print("{:<28}{:>20}{:>28}".format(
        nombre,
        f"({media[0]:.1f}, {media[1]:.1f}, {media[2]:.1f})",
        f"({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f})"
    ))