import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pyautogui
import pydirectinput
import ModuloHandTracking as mh
import threading

y=0
teclaE=False
wCam,hCam = 640,360
mitadWCam= wCam / 2
wScr,hScr = pydirectinput.size()
frameR = 100
smoothing = 10
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0
def mouseMove(handType,fingers,length,lmList):
    global wCam,hCam
    global mitadWCam
    global wScr,hScr
    global frameR
    global smoothing 
    global pLocX, pLocY
    global cLocX, cLocY
    global y
    cX1,cY1 = lmList[0],lmList[1]
    if handType == "Right":
                if int(length) <=19:
                    pydirectinput.mouseDown(button="left")
                else:
                    pydirectinput.mouseUp(button="left")
                if fingers[0] == 0:
                    y=y-15
                    pyautogui.scroll(y)
                elif fingers[0] == 1:
                    y=0
                if fingers[4] == 0:
                    y=y+15
                    pyautogui.scroll(y)
                elif fingers[4] == 1:
                    y=0
                x3 = np.interp(cX1,  (frameR, mitadWCam-frameR), (0, wScr))
                y3 = np.interp(cY1, (frameR, hCam-frameR), (0, hScr))
                print(x3,y3)

                cLocX = pLocX + (x3-pLocX) / smoothing
                cLocY = pLocY + (y3-pLocY) / smoothing
                #print(cLocX,cLocY)
                pydirectinput.moveTo(int(wScr-cLocX), int(cLocY))
     
                pLocX, pLocY = cLocX, cLocY
                print(length)

def main():
    pTime= 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = mh.HandDetector(detectionCon=0.8, maxHands=1)
    while True:
        # Get image frame
        success, img = cap.read()
        #img=cv2.flip(img,1)
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without drawsesee
        print(pydirectinput.position())
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            length, info = detector.findDistance(lmList1[8], lmList1[12])

            fingers1 = detector.fingersUp(hand1)

            mouse = threading.Thread(target=mouseMove,args=(handType1,fingers1,length,lmList1[9]))

            mouse.start()

        # Display
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        #cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('x'): #apretar X para cerrar webcam
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()