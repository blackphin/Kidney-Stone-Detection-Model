import sys
import os

import pandas as pd
import numpy as np
from matplotlib.pyplot import imread
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from sklearn.metrics import f1_score


import gradio as gr


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sns.set_style('darkgrid')
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
print('Modules loaded')


def F1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


model = load_model(r"Model\Model.h5",
                   custom_objects={"F1_score": f1_score})


def recog_model(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (250, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    my_image = imread(img_path)
    prediction = model.predict(x)

    # convert the prediction to a class label
    classes = ['Tumor', 'Cyst', 'Normal', 'Stone']
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = str(100 * (np.max(prediction[0])))
    return (str(predicted_class+" detected with a confidence of "+confidence+"%"))


demo = gr.Interface(fn=recog_model, inputs=gr.Image(image_mode="L", type="filepath", label="Input Image"),
                    outputs=gr.Label(label="Model Prediction"), allow_flagging="never", examples=[r"demo\Cyst.jpg", r"demo\Normal.jpg", r"demo\Stone.jpg", r"demo\Tumor.jpg"])

if __name__ == "__main__":
    demo.launch()
