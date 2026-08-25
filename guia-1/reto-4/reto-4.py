import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

img = "camera"
cv2.namedWindow(img, cv2.WINDOW_NORMAL)

colors = {
    "blue": {
        "low": np.array([100, 60, 40]),
        "high": np.array([140, 255, 230]),
        "color": (255,0,0)
    },
    "yellow": {
        "low": np.array([20, 80, 80]),
        "high": np.array([40, 255, 255]),
        "color": (0, 255, 255)
    },
    "green": {
        "low": np.array([40, 50, 50]),
        "high": np.array([85, 255, 255]),
        "color": (0, 255, 0)
    }
}

loop = True
while loop:
    ret, frame = cap.read()

    if not ret:
        break
   
    # Segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for key, c in colors.items():
        mask = cv2.inRange(hsv, c["low"], c["high"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 400:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x,y), (x+w, y+h), c["color"], 2)
        cv2.putText(frame, key, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c["color"], 2)
    

    # show images
    cv2.imshow(img, frame)

    # check views
    key = cv2.waitKey(10) & 0xFF
    if key == 27 or key == ord('q'):
        loop = False

    if cv2.getWindowProperty(img, cv2.WND_PROP_VISIBLE) < 1:
        loop = False


cap.release()
cv2.destroyAllWindows()
