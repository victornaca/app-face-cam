import cv2
import face_recognition
import pickle
import os

# Importing Peoples Images
folderPath = 'Images'
PathList = os.listdir(folderPath)
imgList = []
peopleIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    peopleIds.append(path.split('.png')[0])
print(peopleIds)

def findEncondings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Enconding Started...")
encodeListKnown = findEncondings(imgList)
encodeListKnownWithIds = [encodeListKnown, peopleIds]
print("Enconding Complete")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")