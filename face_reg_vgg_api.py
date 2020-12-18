import cv2, os, joblib, sys, subprocess 
import numpy as np
from time import time, sleep
from imutils.video import WebcamVideoStream
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains 
from numpy import asarray, expand_dims
from PIL import Image, ImageDraw, ImageFont
from fast_mtcnn import FastMTCNN
from sklearn.preprocessing import Normalizer
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.6,
    keep_all=True,
    device='cuda'
)

embedding_model = VGGFace(
    model='resnet50', 
    include_top=False, 
    input_shape=(224, 224, 3), 
    pooling='avg'
)

def draw_rect(frame, box):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
    x1, y1, x2, y2 = np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def set_label(frame, box, text):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
    x1, y1, x2, y2 = np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2)

    height = np.float32(y2 + 40)

    cv2.rectangle(frame, (x1, y2), (x2, height), (0, 255, 0), cv2.FILLED)

    fontpath = "./font.ttf"
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1+5, y2), text, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)

    return frame

def get_embedding(embedding_model, face):
    pixels = face.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    yhat = embedding_model.predict(samples)

    return yhat[0]

def get_models(embedding_model):
    path = 'models/vgg_models'
    svc_filename, encoder_filename = 'svc_model.mdl', 'encoder_model.mdl'

    encoder_path = os.path.join(path, encoder_filename)
    svc_path = os.path.join(path, svc_filename)

    print('Loading the model from dir...')
    encoder = joblib.load(encoder_path)
    svc_model = joblib.load(svc_path)
    return svc_model, encoder

def get_labels():
    path = 'images/train_data'
    labels = []

    for filename in os.listdir(path):
        label = filename.split('_')[0]
        if label not in labels:
            labels.append(label)

    return labels

def send_name(name):
    subprocess.Popen([sys.executable, 'send_name.py', name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def check_attendance(driver, idx):
    action = ActionChains(driver) 
    radio = driver.find_element_by_id("defaultChecked-" + str(idx))
    driver.execute_script("arguments[0].scrollIntoView(true)", radio)
    driver.execute_script("arguments[0].setAttribute('checked','checked')", radio)
    
    idx_element = driver.find_element_by_id("index-" + str(idx))
    name_element = driver.find_element_by_id("name-" + str(idx))
    attendance_element = driver.find_element_by_id("attendance-" + str(idx))
    comment_element = driver.find_element_by_id("comment-" + str(idx))

    driver.execute_script("arguments[0].setAttribute('class','table-success')", idx_element)
    driver.execute_script("arguments[0].setAttribute('class','table-success')", name_element)
    driver.execute_script("arguments[0].setAttribute('class','table-success')", attendance_element)
    driver.execute_script("arguments[0].setAttribute('class','table-success')", comment_element)

def main():
    svc_model, encoder = get_models(embedding_model)
    print('Completed')

    cap = WebcamVideoStream(src=0).start()

    labels = get_labels()
    driver_path = 'chromedriver.exe'
    url = 'localhost:3000'
    driver = webdriver.Chrome(driver_path)
    driver.get(url)

    start = time()
    count = 0
    fps = 0

    students = []

    while True:
        frame = cap.read()

        if time() - start >= 1:
            fps = count
            count = 0
            start = time()

        count += 1

        boxes, faces = fast_mtcnn([frame])

        if len(faces):
            extracted_faces = []
            
            for face in faces:
                arr = asarray(face)
                if arr.shape[0] > 0 and arr.shape[1] > 0:
                    img = Image.fromarray(face)
                    img_resize = img.resize((224, 224))
                    face_arr = asarray(img_resize)

                    extracted_faces.append(face_arr)

            if len(extracted_faces) == 0:
                continue
            
            embedding_X_train = []

            for face in extracted_faces:
                embedding = get_embedding(embedding_model, face)
                embedding_X_train.append(embedding)

            embedding_X_train = asarray(embedding_X_train)

            norm = Normalizer(norm='l2')
            trainX = norm.transform(embedding_X_train)

            preds = svc_model.predict(trainX)
            pred_probs = svc_model.predict_proba(trainX)
            pred_names = encoder.inverse_transform(preds)

            for pred_prob, pred_name, box in zip(pred_probs, pred_names, boxes[0]):
                accuracy = pred_prob[np.argmax(pred_prob)]*100

                if accuracy > 75:
                    if pred_name not in students:
                        send_name(pred_name)
                        check_attendance(driver, labels.index(pred_name))
                        students.append(pred_name)
                        
                    text = '{} {:.2f}%'.format(pred_name, accuracy)
                else:
                    text = 'Unknown'

                draw_rect(frame, box)
                frame = set_label(frame, box, text)

            for box in boxes[0]:
                draw_rect(frame, box)

        frame = cv2.putText(frame, 'FPS: {}'.format(fps), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Face recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.stop()

            driver.find_element_by_id('confirm').click()
            sleep(5)

            break

if __name__ == "__main__":
    main()