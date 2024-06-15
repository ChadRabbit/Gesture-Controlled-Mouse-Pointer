import cv2
import numpy as np
import pyautogui
import HandTrackingModule as htm
import time
import tensorflow as tf
import mouseinfo
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

wCam,hCam = 640,480
wScr,hScr= 1920,1080

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
imR=150
smoother = 8
plocX,plocY=0,0
clocX,clocY=0,0
pyautogui.FAILSAFE= False
detector=htm.handDetector(maxHands=1)
while True:
    #1
    ret, frame = cap.read()

    frame=detector.findHands(frame)
    lmList,bbox = detector.findPosition(frame)
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        cv2.rectangle(frame,(imR,imR),(wCam-imR,hCam-imR),(255,0,255),2)
        if fingers[1]==1 and fingers[2]==0:
            x3=np.interp(x1,(imR,wCam-imR),(0,wScr))
            y3=np.interp(y1,(imR,hCam-imR),(0,hScr))
            cLocX = plocX+(x3-plocX)/smoother
            cLocY = plocY+(y3-plocY)/smoother
            pyautogui.moveTo(1920-cLocX,cLocY)
            cv2.circle(frame,(x1,y1),15,(0,0,255),cv2.FILLED)
            plocX,plocY=cLocX,cLocY
        if fingers[1]==1 and fingers[2]==1:
            length, img , lineInfo = detector.findDistance(8,12,frame)
            if(length<40):
                cv2.circle(frame,(lineInfo[-2],(lineInfo[-1])),15,(0,255,0),cv2.FILLED)
                pyautogui.click()
                time.sleep(0.5)
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,str(int(fps)),(20,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0 ),3)
    frame = cv2.flip(frame,1)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)
