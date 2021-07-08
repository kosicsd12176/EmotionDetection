import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision import models
device = 'cuda' if torch.cuda.is_available() else 'cpu'



face_classifier = cv2.CascadeClassifier("./dataset/models/haarcascade_frontalface_default.xml")
model_state = torch.load("./dataset/models/emotion_detection_model_state_3.pth")
class_labels = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger", "contempt"]


def get_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 8)
    )
    return model.to(device)

model = get_model()
model.load_state_dict(model_state)

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = Image.fromarray(roi_gray).convert('RGB')
            roi = tt.ToTensor()(roi).unsqueeze(0)

            # make a prediction on the ROI
            tensor = model(roi)
            pred = torch.max(tensor, dim=1)[1].tolist()
            label = class_labels[pred[0]]

            label_position = (x, y)
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (0, 255, 0),
                3,
            )
        else:
            cv2.putText(
                frame,
                "No Face Found",
                (20, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (0, 255, 0),
                3,
            )

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

