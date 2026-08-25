import random
import math
import time

inicio = time.perf_counter()

matriz = []
mini = 255
maxi = 0
suma = 0

for i in range(1000):
    matriz.append([])

    for j in range(1000):
        rng = random.randint(0, 255)

        mini = min(mini, rng)
        maxi = max(maxi, rng)
        suma += rng

        matriz[-1].append(rng)


cantidad = 1000 * 1000

media = suma / cantidad

print("Dimensiones:", len(matriz), "x", len(matriz[0]))
print("Mínimo:", mini)
print("Máximo:", maxi)
print("Media:", media)

# Calcular varianza y desviación estándar
varianza = 0
for fila in matriz:
    for elemento in fila:
        varianza += (elemento - media) ** 2

varianza /= cantidad

desviacion = math.sqrt(varianza)

print("Desviación estándar:", desviacion)


# Aplanar matriz
flatten = []
for fila in matriz:
    for elemento in fila:
        flatten.append(elemento)


# Guardar archivo
with open("output.txt", "w") as f:
    f.write(",".join(map(str, flatten)))
    print("Archivo guardado como output.txt")

fin = time.perf_counter()
print("\nTiempo total:", fin - inicio, "segundos")


# local opencv & numpy
import cv2
import numpy as np

img = np.array(matriz, dtype=np.uint8)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)

while True:
    key = cv2.waitKey(10) & 0xFF

    # ESC o q para salir
    if key == 27 or key == ord('q'):
        break

    # Detectar si cerraste la ventana
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()

