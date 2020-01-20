#importLibraries
from keras.models import load_model
from keras.models import Model
from conf import myConfig as config
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
import PIL.Image as Image
import cv2
import numpy as np
import imageio
from skimage.measure import compare_psnr
import argparse
from pathlib import Path
import keras.backend as K
import tensorflow as tf
#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Set68/',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./model_convolution/v3_100_sigma50.h5',help='pathOfTrainedCNN')
args=parser.parse_args()

#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res
nmodel=load_model(args.weightsPath,custom_objects={'custom_loss':custom_loss})
print('nmodel is loaded')

#createArrayOfTestImages
p=Path('C:/Users/cvlab/Desktop/DBDB_deep/our')
listPaths=list(p.glob('./*.png'))
imgTestArray = []
try:
    for path in listPaths:
        imgTestArray.append(((cv2.resize
        (cv2.imread(str(path),0),(96,96),
        interpolation=cv2.INTER_CUBIC))))
        
except Exception as e:
        print(str(e))
imgTestArray=np.array(imgTestArray)/255
imgTestArray=np.expand_dims(imgTestArray,axis=3)
print("*********",imgTestArray.shape)
p=Path('C:/Users/cvlab/Desktop/DBDB_deep/our')
listPaths=list(p.glob('./*.png'))
imgTest = []
try:
    for path in listPaths:
        imgTest.append(((cv2.resize
        (cv2.imread(str(path),0),(256,256),
        interpolation=cv2.INTER_CUBIC))))
        
except Exception as e:
        print(str(e))
imgTest=np.array(imgTest)/255
imgTest=np.expand_dims(imgTest,axis=3)
#calculatePSNR
sumPSNR=0
noisyImage=imgTestArray+np.random.normal(0.0,50/255,imgTestArray.shape)
noisyImage2=imgTest+np.random.normal(0.0,25/255,imgTest.shape)
print("noisyImage shape:",noisyImage.shape)
noisyImage2=imgTest+np.random.normal(0.0,25/255,imgTest.shape)
for i in range(0,len(imgTestArray)):
    #cv2.imshow('trueCleanImage',imgTestArray[i]); cv2.waitKey(0)
    #noisyImage=imgTestArray[i]+np.random.normal(0.0,25/255,imgTestArray[i].shape)
    #noisyImage=np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0)
    #noisyImage2=imgTest[i]+np.random.normal(0.0,25/255,imgTest[i].shape)
    #noisyImage2=np.expand_dims(np.expand_dims(noisyImage2,axis=2),axis=0)
    #error=nmodel.predict(noisyImage,noisyImage2)
    error,error2,heatmap=nmodel.predict(noisyImage)
    #imgTestArray[i]=np.expand_dims(imgTestArray[i],axis=3)
    print("noisyImage shape:",noisyImage.shape)
    print("imgtestARray[i] shape:",imgTestArray[i].shape)
    print("errorImage shape:",error.shape)
    predClean=noisyImage-error2
    print("predclean shape:",predClean.shape)
    #noisyImage=noisyImage.astype('float32')
    psnr_noise= compare_psnr(imgTestArray[i], noisyImage[i])
    #noisyImage[i] = Image.fromarray((noisyImage[i]*255).astype('uint8'))
    
    #noisyImage[i].save('./result/test_'+str(i+1)+'_sigma'+'25_psnr{:.2f}.png'.format(psnr_noise))
    
    #cv2.imshow('predClean',predClean); cv2.waitKey(0)
        #print(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0).shape)
    
    
        #print(error.min(),error.max())
    #cv2.imshow('predCleanImage',predClean); cv2.waitKey(0)
    #predClean=predClean.astype('float32')
    psnr=compare_psnr(imgTestArray[i],predClean[i])
    psnr2=compare_psnr(imgTestArray[i],noisyImage[i])
    #imageio.imwrite('./result/'+str(i)+'_'+str(psnr2)+'_'+'noisy'+'.png',noisyImage[i])
    #imageio.imwrite('./DnCNN_17664_100_sigma10/'+str(i)+'_'+str(psnr2)+'_'+'noisy'+'.png',noisyImage[i])
    imageio.imwrite('./DnCNN_17664_100_sigma50/'+str(i)+'_'+str(psnr)+'_'+'denoised'+'.png',predClean[i])
    print("predClean shape:",noisyImage[i].shape)
    #noisyImage[i]=noisyImage[i].astype('float32')
    #noisyImage[i] = Image.fromarray((noisyImage[i]*255).astype('uint8'))
    #predClean[i] = Image.fromarray((predClean[i]*255).astype('uint8'))
    #cv2.imwrite("./result/"+str(i)+str(psnr)+".png", imgTestArray[i])
    #predClean[i].save('./DnCNN_17664_100_sigma10/test_'+str(i+1)+'denoised_'+'25_psnr{:.2f}.png'.format(psnr))
    
    print(i,":",psnr)
    sumPSNR=sumPSNR+psnr
    cv2.destroyAllWindows()
avgPSNR=sumPSNR/len(imgTestArray)
print('avgPSNR on test-data',avgPSNR)
print(sumPSNR)

