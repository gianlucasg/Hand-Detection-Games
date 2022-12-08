import cv2
import mediapipe as mp
import time
import numpy as np
import math
import pyautogui
import pydirectinput
from ahk import AHK
ahk = AHK(executable_path='C:/Program Files/AutoHotkey/AutoHotkey.exe')
class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2,complexity = 0, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,model_complexity=self.complexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.DrawingSpec(color=(41,213,37),thickness=2,circle_radius=4),
                mp.solutions.drawing_styles.DrawingSpec(color=(234,130,53),thickness=2,circle_radius=2))
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2):

        x1, y1 = p1[0],p1[1]
        x2, y2 = p2[0],p2[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        return length, info


def main():
    x1=0
    x2=0
    x3=0
    x4=0
    y1=0
    y2=0
    y3=0
    y4=0
    pTime= 0
    cTime = 0
    wCam,hCam = 640,360
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    mano= True
    while True:
        # Get image frame
        success, img = cap.read()
        img=cv2.flip(img,1)
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1)

            if mano:
                if handType1 == "Right":
                    print(fingers1)
                    if fingers1[0] == 0:
                        #print("pulgar abajo")
                        pydirectinput.moveRel(x1,y1,relative=True,_pause=False)
                        #print("x=",x," ","y=",y )
                        x1=x1-1
                    elif fingers1[0] == 1:
                        x1=0
                        y1=0
                    if fingers1[4] == 0:
                        #print("pulgar abajo")
                        pydirectinput.moveRel(x2,y2,relative=True,_pause=False)
                        #print("x=",x," ","y=",y )
                        x2=x2+1
                    elif fingers1[0] == 1:
                        x2=0
                        y2=0
                    if fingers1[1] == 0:
                        #print("pulgar abajo")
                        pydirectinput.moveRel(x3,y3,relative=True,_pause=False)
                        #print("x=",x," ","y=",y )
                        y3=y3-1
                    elif fingers1[1] == 1:
                        x3=0
                        y3=0
                    if fingers1[3] == 0:
                        #print("pulgar abajo")
                        pydirectinput.moveRel(x4,y4,relative=True,_pause=False)
                        #print("x=",x," ","y=",y )
                        y4=y4+1
                    elif fingers1[3] == 1:
                        x4=0
                        y4=0

            # if len(hands) == 2:
            #     # Hand 2
            #     hand2 = hands[1]
            #     lmList2 = hand2["lmList"]  # List of 21 Landmark points
            #     bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            #     centerPoint2 = hand2['center']  # center of the hand cx,cy
            #     handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            #     fingers2 = detector.fingersUp(hand2)
            #     # Find Distance between two Landmarks. Could be same hand or different hands
            #     #length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
            #     # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw

            #         # x=0
            #         # y=0
            #         # while  fingers1[0] == 0:
            #         #     pydirectinput.moveRel(x,y,relative=True,_pause=True)
            #         #     print("x=",x," ","y=",y )
            #         #     x=x+1
            #     if fingers2[0] == 0 or fingers2[1] == 0 or fingers2[3] == 0 or fingers2[4] == 0:
            #         mano = True
            #     if not mano:
            #         if handType2 == "Left":
            #             if fingers2[0] == 0:
            #                 ahk.key_down("w")
            #             elif fingers2[0] == 1:
            #                 ahk.key_up("w")
            #             if fingers2[4] == 0:
            #                 ahk.key_down("s")
            #             elif fingers2[4] == 1:
            #                 ahk.key_up("s")
            #             if fingers2[1] == 0:
            #                 ahk.key_down("d")
            #             elif fingers2[1] == 1:
            #                 ahk.key_up("d")
            #             if fingers2[3] == 0:
            #                 ahk.key_down("a")
            #             elif fingers2[3] == 1:
            #                 ahk.key_up("a")

        # Display
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        #cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('x'): #apretar X para cerrar webcam
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()