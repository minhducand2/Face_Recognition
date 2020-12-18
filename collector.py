import cv2, os
import numpy as np
from os.path import join
from numpy import asarray
from time import sleep
from fast_mtcnn import FastMTCNN

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.6,
    keep_all=True,
    device='cpu'
)

def draw_rect(frame, box):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
    x1, y1, x2, y2 = np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

name = input("Name: ")

cap = cv2.VideoCapture(0)
path = "images/train_data"
iters = 100

print('Start capturing now...')
sleep(1)
i = 1
err = False
isPrintNoFace = False

while i <= 100:
    _, frame = cap.read()

    boxes, faces = fast_mtcnn([frame])

    if len(faces):
        extracted_faces = []
        
        for face in faces:
            arr = asarray(face)
            if arr.shape[0] > 0 and arr.shape[1] > 0:
                extracted_faces.append(face)

        if len(extracted_faces) != 1:
            err = True

    else:
        cv2.imshow("Face collector", frame)
        if not err:
            err = True
            if not isPrintNoFace:
                print('No face detected.')
                isPrintNoFace = True

    if not err:
        filename = r"{}_{}.jpg".format(name, i)
        cv2.imwrite(join(path, 'a.jpg'), frame)
        os.rename(join(path, 'a.jpg'), join(path, filename))

        for box in boxes[0]:
            draw_rect(frame, box)

    cv2.imshow("Face collector", frame)

    if not err:
        print("Captured for image {}".format(i))
        i += 1
        isPrintNoFace = False
    
    err = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.stop()
        break

print("Completed")
