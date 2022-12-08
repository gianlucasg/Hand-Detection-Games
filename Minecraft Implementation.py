import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pyautogui
import pydirectinput
import ModuloHandTracking as mh
import threading

x1=0
x2=0
y1=0
y2=0
teclaE=False
wCam,hCam = 640,360
mitadWCam= wCam / 2
wScr,hScr = pydirectinput.size()
frameR = 130
smoothing = 5
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0
slot=0
slotB=False
def mouseMove(handType,fingers,length,length1):
    global x1
    global x2
    global y1
    global y2
    if handType == "Right":
        #print("mano derecha",fingers)
        #print(length)
        if int(length) <=16:
            pydirectinput.mouseDown(button="left")
        else:
            pydirectinput.mouseUp(button="left")
        if int(length1) <=16:
            pydirectinput.mouseDown(button='middle')
        else:
            pydirectinput.mouseUp(button='middle')
        if fingers[0] == 0:
            #print("pulgar abajo")
            pydirectinput.moveRel(x1,0,relative=True,_pause=False)
            #print("x=",x," ","y=",y )
            x1=x1-1
            #print(x1)
        elif fingers[0] == 1:
            x1=0
        if fingers[4] == 0:
            #print("pulgar abajo")
            pydirectinput.moveRel(x2,0,relative=True,_pause=False)
            #print("x=",x," ","y=",y )
            x2=x2+1
        elif fingers[0] == 1:
            x2=0
        if fingers[1] == 0:
            #print("pulgar abajo")
            pydirectinput.moveRel(0,y1,relative=True,_pause=False)
            #print("x=",x," ","y=",y )
            y1=y1-1
        elif fingers[1] == 1:
            y1=0
        if fingers[3] == 0:
            #print("pulgar abajo")
            pydirectinput.moveRel(0,y2,relative=True,_pause=False)
            #print("x=",x," ","y=",y )
            y2=y2+1
        elif fingers[3] == 1:
            y2=0
def mouseMove1(handType,fingers,length,lmList):
    global wCam,hCam
    global mitadWCam
    global wScr,hScr
    global frameR
    global smoothing 
    global pLocX, pLocY
    global cLocX, cLocY
    global x1
    global x2
    global y1
    global y2
    cX1,cY1 = lmList[0],lmList[1]
    if handType == "Right":
         
                x3 = np.interp(cX1, (frameR, mitadWCam-frameR), (0, wScr))
                y3 = np.interp(cY1, (frameR, hCam-frameR), (0, hScr))
                print(x3,y3)

                cLocX = pLocX + (x3-pLocX) / smoothing
                cLocY = pLocY + (y3-pLocY) / smoothing
                print(cLocX,cLocY)
                pydirectinput.moveTo(int(wScr-cLocX), int(cLocY))
     
                pLocX, pLocY = cLocX, cLocY

def keyboardMove(handType,fingers,length):
    global teclaE
    global slot
    global slotB
    if handType == "Left":
        #print("mano izquierda",fingers)
        if int(length) <=16:
            pydirectinput.mouseDown(button="right")
        else:
            pydirectinput.mouseUp(button="right")
        if fingers[0] == 0:
            pydirectinput.keyDown("w")
        elif fingers[0] == 1:
            pydirectinput.keyUp("w")
        if fingers[1] == 0:
            pydirectinput.keyDown("space")
        elif fingers[1] == 1:
            pydirectinput.keyUp("space")
        if fingers[4] == 0 and not(slotB):
            slotB=True
            if (slot==9):
                slot=1
            else:
                slot=slot+1
            pydirectinput.keyDown(str(slot))
        elif fingers[4] == 1:
            slotB=False
            pydirectinput.keyUp(str(slot))
        if fingers[3] == 0 and not(teclaE):
            teclaE=True
            pydirectinput.keyDown("e")
        elif fingers[3] == 1:
            teclaE=False
            pydirectinput.keyUp("e")
        
        


def main():
    pTime= 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = mh.HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        # Get image frame
        success, img = cap.read()
        #img=cv2.flip(img,1)
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without drawsesee
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right
            length, info = detector.findDistance(lmList1[8], lmList1[12])
            lengthMiddleButon, infoB = detector.findDistance(lmList1[4], lmList1[8])

            fingers1 = detector.fingersUp(hand1)

            mouse = threading.Thread(target=mouseMove,args=(handType1,fingers1,length,lengthMiddleButon))

            mouse.start()

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"
                length, info = detector.findDistance(lmList2[8], lmList2[12])

                fingers2 = detector.fingersUp(hand2)

                keyboard = threading.Thread(target=keyboardMove,args=(handType2,fingers2,length))
                keyboard.start()

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