import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten,Softmax, Add, MaxPooling2D,Input, Dropout
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model_baseline = load_model("./trained_models/cifar10_baseline.h5")
model_scf = load_model("./trained_models/cifar10_scf.h5")



(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

train_images = train_images / 255.
train_labels = train_labels

test_images = test_images / 255.

num_classes=10
train_labels=tf.keras.utils.to_categorical(train_labels,num_classes)
test_labels=tf.keras.utils.to_categorical(test_labels,num_classes)



!pip install cleverhans
import numpy as np
import tensorflow as tf
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method as bim
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method as fgm
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2 as cw

FLAGS = flags.FLAGS
import cv2
import matplotlib.pyplot as plt
import random



m=0
n=0
for i in range(1000):
  score=0
  x=test_images[i]
  y=test_labels[i]
  x=tf.reshape(x,[1,32,32,3])

  adv_img=x
  adv_img=tf.clip_by_value(adv_img,0,1)

  if np.argmax(model_baseline(adv_img))==np.argmax(y):
    m=m+1
    score=(score+model_baseline(adv_img)[0][np.argmax(model_baseline(adv_img))]+model_scf(adv_img)[0][np.argmax(model_baseline(adv_img))])/2
    if score>0.75:
      n=n+1

clean_accuracy=n/m


m=0
n=0
for i in range(1000):
  score=0
  x=test_images[i]
  y=test_labels[i]
  x=tf.reshape(x,[1,32,32,3])
  adv_1=fgm(model_baseline, x,8/255, np.inf)
  adv_noise1=adv_1-x

  adv_img=x+adv_noise1
  adv_img=tf.clip_by_value(adv_img,0,1)

  if np.argmax(model_baseline(x))==np.argmax(y):

    if np.argmax(model_baseline(adv_img))!=np.argmax(y):
      m=m+1
      score=(score+model_baseline(adv_img)[0][np.argmax(model_baseline(adv_img))]+model_scf(adv_img)[0][np.argmax(model_baseline(adv_img))])/2
      if score>0.75:
        n=n+1

zero_knowledge_adversarial_accuracy=1-(n/m)


m=0
n=0
for i in range(1000):
  score=0
  x=test_images[i]
  y=test_labels[i]
  x=tf.reshape(x,[1,32,32,3])
  adv_1=bim(model_baseline, x,8/255,1/255,10, np.inf)
  adv_noise1=adv_1-x
  adv_2=bim(model_scf, x,8/255,1/255,10, np.inf)
  adv_noise2=adv_2-x

  adv_img=x+(1/2)*adv_noise1+(1/2)*adv_noise2
  adv_img=tf.clip_by_value(adv_img,0,1)

  if np.argmax(model_baseline(x))==np.argmax(y):

    if np.argmax(model_baseline(adv_img))!=np.argmax(y):
      m=m+1
      score=(score+model_baseline(adv_img)[0][np.argmax(model_baseline(adv_img))]+model_scf(adv_img)[0][np.argmax(model_baseline(adv_img))])/2
      if score>0.75:
        n=n+1

perfect_knowledge_adversarial_accuracy=1-(n/m)

zero_knowledge_detection_accuracy=(clean_accuracy+zero_knowledge_adversarial_accuracy)/2
perfect_knowledge_detection_accuracy=(clean_accuracy+perfect_knowledge_adversarial_accuracy)/2

print(zero_knowledge_detection_accuracy)
print(perfect_knowledge_detection_accuracy)