import cv2
import pickle
import numpy as np
import cvzone

cap = cv2.VideoCapture('recording.mp4')

with open('CarPosition', 'rb') as f:
    poslist = pickle.load(f)
vac=[]

def checkParking(Fimg):
    Empty= 0
    for i in range(3,len(poslist),4):
        mask = np.zeros(Fimg.shape[0:2], dtype=np.uint8)
        points = np.array([[[poslist[i-3][0],poslist[i-3][1]],[poslist[i-2][0],poslist[i-2][1]],[poslist[i-1][0],poslist[i-1][1]],[poslist[i][0],poslist[i][1]]]])
        #method 1 smooth region
        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        #method 2 not so smooth region
        # cv2.fillPoly(mask, points, (255))
        res = cv2.bitwise_and(Fimg,Fimg,mask = mask)
        rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        ## crate the white background of the same size of original image
        wbg = np.ones_like(Fimg, np.uint8)*255
        cv2.bitwise_not(wbg,wbg, mask=mask)
        # overlap the resulted cropped image on the white background
        """dst = wbg+res
        cv2.imshow('Original',img)
        cv2.imshow("Mask",mask)
        cv2.imshow("Cropped", cropped )
        cv2.imshow("Samed Size Black Image", res)
        cv2.imshow("Samed Size White Image", dst)
        count =  cv2.countNonZero(cropped)"""
        number_of_white_pix= np.sum(cropped == 255)
        number_of_black_pix = np.sum(cropped == 0)
        total=number_of_white_pix+number_of_black_pix
        count=(number_of_white_pix/total)*100

        if count <10:
            color = (0, 255, 0)
            Empty+= 1
            vac.append(i)
        else:
            color = (0, 0, 255)
        pts = np.array([list(poslist[i-3]),list(poslist[i-2]),list(poslist[i-1]),list(poslist[i])],np.int32)
        pts = pts.reshape((-1, 1, 2))
        #cv2.rectangle(img, (poslist[i-1][0],poslist[i-1][1]), (poslist[i][0], poslist[i][1]), (255, 0, 255), 2)
        cv2.polylines(img, [pts], True, color,2)
        cvzone.putTextRect(img, str(count), (poslist[i-3][0], poslist[i-1][1] - 3), scale=1,
                           thickness=2, offset=0, colorR=color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale =0.7
    color = (0, 0, 255)
    thickness = 1
    cv2.putText(img,  f'Vaccant: {Empty}/{len(poslist)/4}', (20, 20), font,
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, f'{vac}', (30, 520), font,
                   fontScale, color, thickness, cv2.LINE_AA)
   
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5 ), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParking(imgDilate)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
 
# Using resizeWindow()
    cv2.resizeWindow("Resized_Window", 900, 600)
 
# Displaying the image
    cv2.imshow("Resized_Window", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
