from keras import backend as K
import keras
import cv2
from Utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import *
from keras.applications import MobileNetV2
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os
import glob

# In[3]:


# Variable Definition
img_w = 512
img_h = 512
channels = 3
classes = 1
info = 5
grid_w = 16
grid_h = 16
cropped = None

def load_model(strr):
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


def yolo_model(input_shape):
    inp = Input(input_shape)

    model = MobileNetV2(input_tensor=inp, include_top=False, weights='imagenet')
    last_layer = model.output

    conv = Conv2D(512, (3, 3), activation='relu', padding='same')(last_layer)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)

    conv = Conv2D(128, (3, 3), activation='relu', padding='same')(lr)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)

    conv = Conv2D(5, (3, 3), activation='relu', padding='same')(lr)

    final = Reshape((grid_h, grid_w, classes, info))(conv)

    model = Model(inp, final)

    return model


# In[10]:


# define loss function
def yolo_loss_func(y_true, y_pred):
    # y_true : 16,16,1,5
    # y_pred : 16,16,1,5
    l_coords = 5.0
    l_noob = 0.5
    coords = y_true[:, :, :, :, 0] * l_coords
    noobs = (-1 * (y_true[:, :, :, :, 0] - 1) * l_noob)
    p_pred = y_pred[:, :, :, :, 0]
    p_true = y_true[:, :, :, :, 0]
    x_true = y_true[:, :, :, :, 1]
    x_pred = y_pred[:, :, :, :, 1]
    yy_true = y_true[:, :, :, :, 2]
    yy_pred = y_pred[:, :, :, :, 2]
    w_true = y_true[:, :, :, :, 3]
    w_pred = y_pred[:, :, :, :, 3]
    h_true = y_true[:, :, :, :, 4]
    h_pred = y_pred[:, :, :, :, 4]

    p_loss_absent = K.sum(K.square(p_pred - p_true) * noobs)
    p_loss_present = K.sum(K.square(p_pred - p_true))
    x_loss = K.sum(K.square(x_pred - x_true) * coords)
    yy_loss = K.sum(K.square(yy_pred - yy_true) * coords)
    xy_loss = x_loss + yy_loss
    w_loss = K.sum(K.square(K.sqrt(w_pred) - K.sqrt(w_true)) * coords)
    h_loss = K.sum(K.square(K.sqrt(h_pred) - K.sqrt(h_true)) * coords)
    wh_loss = w_loss + h_loss

    loss = p_loss_absent + p_loss_present + xy_loss + wh_loss

    return loss


# In[11]:


model = load_model('model/text_detect_model.json')
model.load_weights('model/text_detect.h5')


# In[16]:


def predict_func(model, inp, iou, name):
    global cropped
    ans = model.predict(inp)

    # np.save('Results/ans.npy',ans)
    boxes = decode(ans[0], img_w, img_h, iou)

    img = ((inp + 1) / 2)
    img = img[0]
    # plt.imshow(img)
    # plt.show()

    for i in boxes:
        i = [int(x) for x in i]

        img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), color=(0, 255, 0), thickness=2)
        print('x',i[0],'y', i[1],'w', i[2],'h', i[3])
        y1, y2, x1, x2 = i[1], i[3], i[0], i[2]
        cropped = img[y1:y2, x1:x2]
        cv2.imshow(str(i[0]), cropped)
    plt.imshow(img)
    plt.show()

    cv2.imwrite(os.path.join('Results', str(name) + '.jpg'), img * 255.0)


# In[17]:

for i in os.listdir('Test'):
    img = cv2.imread(os.path.join('Test', i))
    img = cv2.resize(img, (512, 512))
    img = (img - 127.5) / 127.5
    predict_func(model, np.expand_dims(img, axis=0), 0.5, 'sample')

# ### THE END
