import pickle
import numpy as np
import cv2
import face_recognition
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Load Enconding File
print("Loading Enconding File...")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
encodeListKnown, peopleIds = encodeListKnownWithIds
print("Enconding File Loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        # print("Match Index", matchIndex)

        if matches[matchIndex]:
            print("Known Face Detected")
            print(peopleIds[matchIndex])
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            bbox = x1, y1, x2-x1, y2-y1
            img = cvzone.cornerRect(img,bbox,rt=0)

    cv2.imshow("Face Attendance", img)
    cv2.waitKey(1)