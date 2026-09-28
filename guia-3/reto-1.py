import os
import random
import cv2
import xml.etree.ElementTree as ET
import numpy as np

dataset_base_path = os.path.join(os.path.expanduser('~') ,"Downloads/MASATI-v2")
masati_content = os.listdir(dataset_base_path)
with_labels = {}

# find folders with labels
for folder in masati_content: 
    if folder.endswith("_labels"):
        with_labels[folder.replace("_labels", "")] = folder

# Escoger aleatoriamente imagenes y mostrarlas con rectangulos en los barcos
for data, label in with_labels.items():
    data_path = os.path.join(dataset_base_path, data)
    label_path = os.path.join(dataset_base_path, label)
    image = random.choice(os.listdir(data_path))

    tree = ET.parse(os.path.join(label_path, image.replace(".png", ".xml")))
    root = tree.getroot()
    rectangles = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        rectangles.append((xmin, ymin, xmax, ymax))

    cv2_img = cv2.imread(os.path.join(data_path, image))
    img = cv2_img.copy()
    for (x1,y1,x2,y2) in rectangles:
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,0,255), 1)

    cv2.imshow(image, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
