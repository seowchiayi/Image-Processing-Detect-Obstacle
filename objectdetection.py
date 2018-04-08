import cv2
import numpy as np

def preprocess_image(frame):
    #150,5,233
    pink = np.uint8([[[147,20,255]]])
    hsv_pink = cv2.cvtColor(pink,cv2.COLOR_BGR2HSV)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of pink color in HSV may try with +10 or -10
    lowerBound=np.array([hsv_pink[0][0][0]-20,147,147])
    upperBound=np.array([hsv_pink[0][0][0]+20,255,255])

    # Threshold the HSV image to get only pink colors
    mask = cv2.inRange(hsv, lowerBound, upperBound)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask = mask)

    #morphology
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
    finalResult = closing.copy()
    finalResult = cv2.cvtColor(finalResult,cv2.COLOR_RGB2GRAY)

    return finalResult

def get_refer_point(ret,frame):
# while(cap.isOpened()):
#     ret,frame = cap.read()
    # if ret is True:
        finalResult = preprocess_image(frame)
        img,cont,h= cv2.findContours(finalResult.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        d = {}
        rfx = 0
        rfy = 0
        if (len(cont)!=0):
            for i in range(len(cont)):
                area = cv2.contourArea(cont[i])
                if(area>8000):
                    cv2.drawContours(frame,cont[i],contourIdx = -1,color = (255,0,0),thickness = 2,maxLevel = 1)
                    x,y,w,h = cv2.boundingRect(cont[i])

                    f = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(f,"Obstacle",(x+10,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
                    approx = cv2.approxPolyDP(cont[i],0.05*cv2.arcLength(cont[i],True),True)
                    ry_lst = []
                    bottom_corners = []
                    for a in range(len(approx)):
                        rx = approx[a][0][0]
                        ry = approx[a][0][1]
                        ry_lst.append(approx[a][0])

                        #bottom_corners =
                        cv2.circle(f,(rx,ry),5,0,-1)
                        cv2.putText(f,('(' + str(rx) + ',' + str(ry) + ')'),(rx-40,ry+15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,True)

                    ry_lst = sorted(ry_lst, key = lambda k: k[1])
                    bottom_corners.append(ry_lst[-2:])
                    refer_point = [(bottom_corners[0][1][0] + bottom_corners[0][0][0])/2, (bottom_corners[0][1][1] + bottom_corners[0][0][1])/2]
                    rfx = int(refer_point[0])
                    rfy = int(refer_point[1])
                    cv2.circle(f,(rfx,rfy),5,(0,255,255),-1)
                    cv2.putText(f,('(' + str(rfx) + ',' + str(rfy) + ')'),(rfx-40,rfy+30),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,True)
                d[i] = {'Area':area,'Refer_point':[rfx,rfy]}
                return d
        else:
            return ('No Obstacle Found')


        # cv2.imshow('showContour',f)
        # k = cv2.waitKey(350) & 0xFF
        # if k == 27:
        #     break

cap = cv2.VideoCapture('robotics2_obstacle.mp4')
while(cap.isOpened()):
    # Take each frame
    ret,frame = cap.read()
    #time.sleep(0.1)
    if ret is True:
        dict_of_info = get_refer_point(ret,frame)
        print(dict_of_info)
    else:
        print ('Failed to read frame')
        break

cap.release()
cv2.destroyAllWindows()
