import pickle
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://app-face-cam-default-rtdb.firebaseio.com/",
    'storageBucket':"app-face-cam.appspot.com"
})

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load Encoding File
print("Loading Encoding File...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
encodeListKnown, peopleIds = encodeListKnownWithIds
print("Encoding File Loaded")

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            # Draw rectangle around the face
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            id = peopleIds[matchIndex]
            peopleInfo = db.reference(f'Peoples/{id}').get()

            # Display your Name
            cv2.rectangle(img, (left, top - 30), (right, top-5), (33,94,33), cv2.FILLED)
            cv2.putText(img, str(peopleInfo['name']), (left+5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, str(peopleInfo['type']), (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33,94,33), 2)
            print(peopleInfo)

    cv2.imshow("Security Cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
