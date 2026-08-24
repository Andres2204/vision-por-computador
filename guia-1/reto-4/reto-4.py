import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

images = ["camera", "cv2"]

for img in images:
    cv2.namedWindow(img, cv2.WINDOW_NORMAL)

loop = True
while loop:
    ret, frame = cap.read()

    if not ret:
        break
    
    # Segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    low_blue = np.array([100, 60, 40])
    high_blue = np.array([140, 255, 230])

    blue_mask = cv2.inRange(hsv, low_blue, high_blue) 
    
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    sorted_contours_desc = sorted(contours, key=cv2.contourArea, reverse=True)

    cv2.drawContours(frame, sorted_contours_desc[0:4], -1, (0,0,255), 1, thickness=2)


    # show images
    cv2.imshow(images[0], frame)
    cv2.imshow(images[1], blue_mask)
    #cv2.imshow(images[1], threshold)

    # check views
    key = cv2.waitKey(10) & 0xFF
    if key == 27 or key == ord('q'):
        loop = False

    for img in images:
        if cv2.getWindowProperty(images[0], cv2.WND_PROP_VISIBLE) < 1:
            loop = False


cap.release()
cv2.destroyAllWindows()
