import random
import math

matriz = []
mini = 255
maxi = 0
media = 0

for i in range(0, 1000):
    matriz.append([])
    for j in range(0, 1000):
        rng = random.randint(0, 255)
        mini = min(mini, rng)
        maxi = max(maxi, rng)
        if media != 0: 
            media = (media + rng)/2
        else:
            media = rng
        matriz[-1].append(rng)


print("Dimenciones: " + str(len(matriz)) + "x" + str(len(matriz[0])))
print("mini: " + str(mini) + " maxi: " + str(maxi) + " media: " + str(media))

varianza = 0
for m in matriz:
    for e in m: 
        varianza += e**2

varianza /= 1000*1000
desviacion = math.sqrt(varianza);
print("desviacion estandar: " + str(desviacion))

flatten = sum(matriz, [])
with open("output.txt", "w") as f:
    f.write(str(flatten))
    print("archivo guardado como output.txt")

"""
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
"""
