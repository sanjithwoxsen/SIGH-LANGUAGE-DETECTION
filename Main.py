import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import streamlit as st
from gtts import gTTS
import os

def Speech(mytext):

    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=mytext, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")

    # Playing the converted file
    os.system("welcome.mp3")


def HandSignDetect():
    # Capture and initializing values
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    offset = 450
    offset1 = 20
    imgSize = 1000
    labels = ["HI", "Please", "I LOVE YOU","MY","Thank You","Yes","Time"]
    counter = 0
    Text =""
    previous_text = ""
    Final_text = ""
    previous_word =""


    # Streamlit Setup
    frame_holder = st.empty()
    stop_button = st.button("Stop")
    st.subheader("Text :")
    session1 = st.empty()
    while cap.isOpened() and not stop_button:
        try:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:

                hand1 = hands[0]
                lmList1 = hand1["lmList"]
                x, y, w, h = hand1["bbox"]
                cx, cy = hand1["center"]
                centerPoint1 = cx, cy
                handType1 = hand1["type"]
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset1:y + h + offset1, x - offset1:x + w + offset1]
                imgCropShape = imgCrop.shape

                fingers1 = detector.fingersUp(hand1)

                if len(hands) == 1:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        Text = labels[index]
                        print(Text)

                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        Text = labels[index]
                        print(Text)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                    (255, 255, 255),
                                    2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

                if len(hands) == 2:
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"]
                    x, y, w, h = hand2["bbox"]
                    cx1, cy1 = hand2["center"]
                    centerPoint2 = cx1, cy1
                    px = (cx1 + cx) // 2
                    py = (cy1 + cy) // 2
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgCrop = img[py - offset:py + offset, x - offset:px + offset]

                    imgCropShape = imgCrop.shape

                    handType2 = hand2["type"]

                    fingers2 = detector.fingersUp(hand2)

                    length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        Text = labels[index]
                        print(Text)
                        print(labels[index])


                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        Text = labels[index]
                        print(Text)
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255),
                                    2)


                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                      (x + w + offset, y + h + offset), (255, 0, 255), 4)


                if Text == previous_text:
                    counter += 1

                    if counter == 6 and previous_word != Text :
                        # Entry is allowed
                        print("Entry accepted.")
                        Final_text = Final_text+" "+ f"{Text}"
                        previous_word = Text
                        counter = 0
                        Speech(Text)
                        print(Final_text)
                else:
                    counter = 0
                    previous_text = Text



            Frame = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
            frame_holder.image(Frame, channels="RGB")






        except Exception as a:
            print(f"An exception {a} occurred")


        session1.text(Final_text)
    else:
        cap.release()



#Title name setup
st.title("Hand Sign Translator")

session = st.empty()

start_btn = session.button("Start")
if start_btn:
    session.empty()
    HandSignDetect()


