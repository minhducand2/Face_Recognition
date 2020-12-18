import os, joblib
from numpy import asarray, expand_dims
from PIL import Image
from fast_mtcnn import FastMTCNN
from sklearn.preprocessing import Normalizer, LabelEncoder
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.svm import SVC

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

full_labels = []

def getTrainDataFromDir(size=(224, 224)):
    path = 'images/train_data'
    extracted_faces = []
    labels = []

    for filename in os.listdir(path):
        img = Image.open(f'{path}/{filename}')
        img = img.convert('RGB')
        img_arr = asarray(img)
        boxes, faces = fast_mtcnn([img_arr])

        face = faces[0]
        img = Image.fromarray(face)
        img_resize = img.resize(size)
        face_arr = asarray(img_resize)

        extracted_faces.append(face_arr)

        label = filename.split('_')[0]
        labels.append(label)  
        if label not in full_labels:
            full_labels.append(label)

    return asarray(extracted_faces), asarray(labels)

def get_embedding(embedding_model, face):
    pixels = face.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    yhat = embedding_model.predict(samples)

    return yhat[0]

def train(embedding_model):
    X_train, y_train = getTrainDataFromDir()
    embedding_X_train = list()
    path = 'models/vgg_models'
    svc_filename, encoder_filename = 'svc_model.mdl', 'encoder_model.mdl'

    encoder_path = os.path.join(path, encoder_filename)
    svc_path = os.path.join(path, svc_filename)

    for face in X_train:
        embedding = get_embedding(embedding_model, face)
        embedding_X_train.append(embedding)

    embedding_X_train = asarray(embedding_X_train)

    norm = Normalizer(norm='l2')
    trainX = norm.transform(embedding_X_train)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    trainy = encoder.transform(y_train)

    print('Training the model...')

    svc_model = SVC(kernel='sigmoid', probability=True)
    svc_model.fit(trainX, trainy)

    joblib.dump(svc_model, svc_path)
    joblib.dump(encoder, encoder_path)

def main():
    train(embedding_model)
    print('Training completed.')

if __name__ == "__main__":
    main()