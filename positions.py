import cv2
import pickle
import numpy as np
try:
    with open('CarPosition', 'rb') as f:
        poslist = pickle.load(f)
except:
    poslist = []
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        poslist.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i in range(3,len(poslist),4):
            if poslist[i-3][0] < x < poslist[i-1][0] and poslist[i-3][1] < y < poslist[i-1][1]:
                poslist.pop(i)
                poslist.pop(i-1)
                poslist.pop(i-2)
                poslist.pop(i-3)
    with open('CarPosition', 'wb') as f:
        pickle.dump(poslist, f)
while True:
    img = cv2.imread('rframe.jpg')
    for i in range(3,len(poslist),4):
       
        pts = np.array([list(poslist[i-3]),list(poslist[i-2]),list(poslist[i-1]),list(poslist[i])],np.int32)
        pts = pts.reshape((-1, 1, 2))
        #cv2.rectangle(img, (poslist[i-1][0],poslist[i-1][1]), (poslist[i][0], poslist[i][1]), (255, 0, 255), 2)
        cv2.polylines(img, [pts], True, (0,255,100),2)

    #cv2.rectangle(img, (85,70), (187,115), (255, 0, 255), 2)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
 
# Using resizeWindow()
    cv2.resizeWindow("Resized_Window", 900, 600)
 
# Displaying the image
    cv2.imshow("Resized_Window", img)
    cv2.setMouseCallback("Resized_Window", mouseClick)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    #cv2.waitKey(1)
cv2.destroyAllWindows()