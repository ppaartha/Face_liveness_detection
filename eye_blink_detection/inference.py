import os
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from PIL import Image
import time
# from scipy.misc import imresize, imsave

IMG_SIZE = 24

def load_pretrained_model():
    model = load_model('eye_status_classifier.h5')
    # model.summary()
    return model

def init():
    face_cascPath = 'haarcascade_frontalface_alt.xml'
    # face_cascPath = 'lbpcascade_frontalface.xml'

    open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
    right_eye_cascPath ='haarcascade_righteye_2splits.xml'
    dataset = r'D:\face_recognition\Anti-Spoofing-in-Face-Recognition-master\dataset\faces'

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    model = load_pretrained_model()


    print("[LOG] Collecting images ...")
    images = []
    for direc, _, files in tqdm(os.walk(dataset)):
        for file in files:
            if file.endswith("jpeg"):
                images.append(os.path.join(direc,file))
    return (model,face_detector, open_eyes_detector, left_eye_detector,right_eye_detector, images)

def process_and_encode(images):
    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []
    print("[LOG] Encoding faces ...")

    for image_path in tqdm(images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
        boxes = face_recognition.face_locations(image, model='cnn')

        # Encode the face into a 128-d embeddings vector
        encoding = face_recognition.face_encodings(image, boxes)

        # the person's name is the name of the folder where the image comes from
        name = image_path.split(os.path.sep)[-2]

        if len(encoding) > 0 : 
            known_encodings.append(encoding[0])
            known_names.append(name)

        encodings = {"encodings": known_encodings, "names": known_names}
        np.save('encodings.npy', encodings) 


    return encodings


def predict(img, model):
    img = Image.fromarray(img, 'RGB').convert('L')
    img = np.array(img)  # Convert PIL image to numpy array
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype('float32')
    img /= 255
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    print("Prediction",prediction)
    
    if prediction < 0.1:
        prediction = 'closed'
    elif prediction > 0.9:
        prediction = 'open'
    else:
        prediction = 'idk'
    
    return prediction

def evaluate(X_test, y_test):
	model = load_model()
	print('Evaluate model')
	loss, acc = model.evaluate(X_test, y_test, verbose = 0)
	print(acc * 100)

def isBlinking(history, maxFrames):
    """ 
    @history: A string containing the history of eyes status 
              where a '1' means that the eyes were closed and '0' open.
    @maxFrames: The maximal number of successive frames where an eye is closed 
    """
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, max_blink_frames=10):
    frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    eyes_detected = ""
    
    # For each detected face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]

        # Eyes detection
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(open_eyes_glasses) == 2:
            eyes_detected += '1'
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            left_face = frame[y:y+h, x+int(w/2):x+w]
            left_face_gray = gray[y:y+h, x+int(w/2):x+w]

            right_face = frame[y:y+h, x:x+int(w/2)]
            right_face_gray = gray[y:y+h, x:x+int(w/2)]

            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            eye_status = '1'

            for (ex, ey, ew, eh) in right_eye:
                color = (0, 255, 0)
                pred = predict(right_face[ey:ey+eh, ex:ex+ew], model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(right_face, (ex, ey), (ex+ew, ey+eh), color, 2)

            for (ex, ey, ew, eh) in left_eye:
                color = (0, 255, 0)
                pred = predict(left_face[ey:ey+eh, ex:ex+ew], model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(left_face, (ex, ey), (ex+ew, ey+eh), color, 2)

            eyes_detected += eye_status

        if isBlinking(eyes_detected, max_blink_frames):
            # print("YES")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, 'Real', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            # print("NO")
            if len(eyes_detected) > 20:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, 'Fake', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame

(model, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, images) = init()


# print("Images: ",images)

# data = process_and_encode(images)
data = np.load('encodings.npy',allow_pickle='TRUE').item()

# print(data)


print("[LOG] Opening webcam ...")
video_capture = VideoStream(src=0).start()

eyes_detected = defaultdict(str)
while True:
    frame = detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector)
    cv2.imshow("Eye-Blink based Liveness Detection for Facial Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_capture.stop()