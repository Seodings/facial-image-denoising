# import packages
#from conf import myConfig as config
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Add, Flatten, Dense ,Dropout,MaxPooling2D,Reshape,Subtract,UpSampling2D,Conv2DTranspose
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Input
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from pathlib import Path
import tensorflow as tf
import keras.backend as K
import argparse
import numpy as np
import keras
import pandas as pd
import cv2
def Dncnn(x):
    x = Conv2D(filters=64, kernel_size=(3,3),padding='same')(x)
    x = Activation('relu')(x)
    for layers in range(2,11):
        x = Conv2D(filters=64, kernel_size=(3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Conv2D(filters=1, kernel_size=(3,3),padding='same')(x)
    return x


# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*32) #2*batchsize
    return res
def Facial(x):
    x = Conv2D(filters=32,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x_facial", tf.shape(x))
    
    x = Conv2D(filters=64,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x_facial", tf.shape(x))
    
    x = Conv2D(filters=96,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=96,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x_facial", tf.shape(x))

    x = Conv2D(filters=128,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x_facial", tf.shape(x))

    x = Conv2D(filters=256,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("x_facial", tf.shape(x))

    x = Conv2D(filters=512,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512,kernel_size=(3,3), padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x) #(,3,3,512)
    print("x_facial", tf.shape(x))
    x = Conv2D(filters=30,kernel_size=(2,2), strides=2, padding='same', use_bias=False)(x) #( ,1,1,30)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=30,kernel_size=(2,2), strides=2, padding='same', use_bias=False)(x) #( ,1,1,30)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    return x
def Encoder(x):
    x = Conv2D(filters=32,kernel_size=(3,3), strides=2, use_bias=False)(x) #( ,23,23,30)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64,kernel_size=(3,3), strides=2, use_bias=False)(x) #( ,11,11,30)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128,kernel_size=(3,3), strides=2, use_bias=False)(x) #( ,5,5,30)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128,kernel_size=(3,3), strides=2, use_bias=False)(x) #( ,2,2,30)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64,kernel_size=(2,2), strides=2, use_bias=False)(x) #( ,1,1,30)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=30,kernel_size=(2,2), strides=2, use_bias=False)(x) #( ,1,1,30)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def Decoder(x):
    x = Conv2DTranspose(filters=30,kernel_size=(3,3),strides=2,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) #(,2,2,1)
    x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) #(,5,5,1)
    x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) #(,11,11,1)
    x = Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) #(,23,23,1)
    x = Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) #(,48,48,1)
    x = Conv2DTranspose(filters=1,kernel_size=(4,4),strides=2,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) #(,48,48,1)
    return x
if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    inputs = Input(shape=(96,96,1))
    dncnn = Dncnn(inputs) #( ,48,48,1)
    dncnns = Subtract()([inputs,dncnn]) #(,48,48,1)
    #detectioninput = UpSampling2D((2, 2))(dncnns)#(96,96,1)
    print("dncnns shape:",dncnns.shape)
    encoder = Encoder(dncnns) #(,30)
    #detectioninput = Input(shape=(96,96,1))
    detection = Facial(dncnns) #( ,1,1,30)
    print("encoder shape:,",encoder.shape)
    print("detection shape:,",detection.shape)
    x = Add()([encoder, detection]) #(, 1,1,30)
    print("x shape",x.shape)
    decoder = Decoder(x)
    print("decoder:",decoder.shape) #(,48,48,1)
    outputss = Dncnn(decoder)
    print("ouputss:",outputss.shape)
    output =Add()([dncnn, outputss])
    

    
    model = keras.Model(inputs=[inputs],outputs=[dncnn,output,detection])
    model.summary()
    opt=optimizers.Adam(lr=0.001)
    model.compile(loss=custom_loss,optimizer=opt)
    
    #load the data and normalize it
    '''
    cleanImages=np.load('C:/Users/cvlab/Desktop/DBDB_deep/dataset_npy/train_17664_48.npy')
    cleanImages = cleanImages.reshape(-1,cleanImages.shape[1],cleanImages.shape[2],1)
    cleanImages=cleanImages/255.0
    cleanImages=cleanImages.astype('float32')
    #cleanImages=np.expand_dims(cleanImages,axis=3)
    
    
    truenoiseImages = np.random.normal(0.0,10/255.0,cleanImages.shape)
    noiseImages = cleanImages+truenoiseImages
    '''
    p=Path('C:/Users/cvlab/Desktop/DBDB_deep/dataset/train_17664_48')
    listPaths=list(p.glob('./*.png'))
    cleanImages = []
    try:
        for path in listPaths:
            cleanImages.append(((cv2.resize
            (cv2.imread(str(path),0),(96,96),
            interpolation=cv2.INTER_CUBIC))))
            
    except Exception as e:
            print(str(e))
    cleanImages=np.array(cleanImages)/255
    cleanImages=np.expand_dims(cleanImages,axis=3)
    truenoiseImages = np.random.normal(0.0,25/255.0,cleanImages.shape)
    noiseImages = cleanImages+truenoiseImages
    print("*********",cleanImages.shape)
    
    print("noisy image shape:",noiseImages.shape)
   # cv2.imshow('cleanImages[0]',cleanImages[0]); cv2.waitKey(0)
    #cv2.imshow('noiseImages[0]',noiseImages[0]); cv2.waitKey(0)
    y_train = np.load('C:/Users/cvlab/Desktop/DBDB_deep/dataset_npy/train_17664_96_heatmap.npy')
    y_train = y_train.reshape(-1,1,1,30)
    #y_train = y_train/255.0
    #y_train = y_train.astype('float32')

    print("y_train shape:",y_train.shape) #(17664,1,1,30)
    
    
    # define augmentor and create custom flow
    aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")
    callbacks=[LearningRateScheduler(lr_decay)]
    # tr;ain

    model.fit([noiseImages], [truenoiseImages,truenoiseImages,y_train],epochs=100,callbacks=callbacks,verbose=1)

# save the model
    model.save('./model_convolution/v3_100_sigma25.h5')
