import cv2
import numpy as np
import time
print (cv2.__version__)
cap = cv2.VideoCapture('robotics2_obstacle.mp4')

pink = np.uint8([[[150,5,233]]])
hsv_pink = cv2.cvtColor(pink,cv2.COLOR_BGR2HSV)

while(cap.isOpened()):
    # Take each frame
    ret,frame = cap.read()
    #time.sleep(0.1)
    if ret is True:

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of pink color in HSV may try with +10 or -10
        lowerBound=np.array([hsv_pink[0][0][0]-17,155,155])
        upperBound=np.array([hsv_pink[0][0][0]+17,255,255])

        # Threshold the HSV image to get only pink colors
        mask = cv2.inRange(hsv, lowerBound, upperBound)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        #morphology
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)
        closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
        finalResult = closing.copy()
        finalResult = cv2.cvtColor(finalResult,cv2.COLOR_RGB2GRAY)
        img,cont,h= cv2.findContours(finalResult.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for i in range(len(cont)):
            area = cv2.contourArea(cont[i])
            if(area>2000):
                cv2.drawContours(frame,cont[i],-1,(255,0,0),2)
                x,y,w,h = cv2.boundingRect(cont[i])
                f = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(f,"Obstacle",(x+10,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
                #cv2.putText(f,('(' + str(x) + ',' + str(y) + ')'),(x-20,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)
            cv2.imshow('frame',frame)
            cv2.imshow('mask',mask)
            cv2.imshow('res',res)
            cv2.imshow('closing',closing)
            cv2.imshow('showContour',f)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
