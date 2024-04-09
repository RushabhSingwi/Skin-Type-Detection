import os
import cv2
import time
import FaceDetectionModule as ftm

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = ftm.FaceDetector(cascade_path)

dir = '/Users/rushabh/Code/AI/Skin/RealTimeDetections'

for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

path = "/Users/rushabh/Code/AI/Skin/assets/oil-vid.mp4"
cap = cv2.VideoCapture(path)
expTime = 0
while True:
    ret, img = cap.read()
    img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
    calTime = time.time()

    curTime = time.time()
    expTime = curTime - expTime
    expTime = curTime
    if expTime > 3:
        filename = str(int(expTime))
        img, bboxs = detector.find_faces(img)
        x, y, w, h = bboxs[0]
        img_pred = img[y:y + h, x:x + w]
        img_pred = cv2.resize(img_pred, (224, 224))
        cv2.imwrite(f"./RealTimeDetections/{filename}.jpg", img_pred)
        break

def getImg():
    dir = os.listdir("./RealTimeDetections")

    return  dir[0]