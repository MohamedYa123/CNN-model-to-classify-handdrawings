import cv2
im = cv2.imread("images/Image3.png")
import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
#im=im/255.0
#cv2.imshow('Original', im) 
#cv2.waitKey(0) 
#print(im/255.0)
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
resized_image = cv2.resize(gray_image, (28, 28))
resized_image=np.reshape(resized_image,(28,28,1))
#cv2.imshow('Original', gray_image) 
#cv2.waitKey(0) 
#cv2.imshow(' ',resized_image)
#cv2.waitKey(0)
filehandler=open('minMaxScaler.pickle','rb')
minMaxScaler=pickle.load(filehandler)
resized_image=resized_image/255.0

#print(resized_image)
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
tlen=12
'''

model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(1, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.UpSampling2D((2, 2))
'''
#tlen=3
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(tlen*2, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(tlen, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
#model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(8,activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
#model.add(tf.keras.layers.Dense(8,activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
#model.add(tf.keras.layers.Dense(7,activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.Dense(tlen,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy','sparse_categorical_crossentropy'])
model.summary()

dtrainx=np.array([resized_image])
s=(resized_image.shape[0]*resized_image.shape[1])
s2=(resized_image.shape[0],resized_image.shape[1])
resized_image=np.reshape(resized_image,s)
dtrainx2=np.array([resized_image])
#dtrainx2=minMaxScaler.transform(dtrainx2)
dtrainx=np.reshape(dtrainx2,(len(dtrainx2),s2[0],s2[1],1))
#dtrainx=np.array([resized_image])
print(dtrainx.shape)
print(resized_image.shape)
#dtrainx=minMaxScaler.transform(dtrainx)
dtrainy=np.array(np.array([[0.]]))
print(dtrainy)
model.fit(dtrainx,dtrainy,epochs=1)

outp=[' airplane', ' alarm clock', ' ambulance', ' ant', ' axe', 
      ' banana', ' bandage', ' basketball', ' bear', ' bed', 
      ' bicycle', ' The Eiffel Tower']





model.load_weights('best_weightsFinal.keras')
a=model.predict(dtrainx)[0]
print(a)
print(a[np.argmax(a)])
title=outp[np.argmax(a)]
plt.title(title)
print(im.shape)
im2=np.array(im)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        #k=im2[i,j]
        if(im2[i,j,0]<20 and im2[i,j,1]<20 and im2[i,j,2]<20):
                jh=[[i+1,j+1],[i-1,j-1],[i-1,j],[i,j-1],[i+1,j],[i,j+1]]
                for ah in jh:
                     i2=ah[0]
                     j2=ah[1]
                     if(i2<im.shape[0] and j2<im.shape[1] and i2>-1 and j2>-1): 
                        z1=im2[i2,j2,0]
                        z2=im2[i2,j2,1]
                        z3=im2[i2,j2,2]
                        if z1>0 and z2>0 and z3>0:
                            # print(z1,z2,z3)
                             im2[i2,j2,1]=255
#plt.show()                              
plt.imshow(im2)
plt.figure()
plt.show()
plt.title('Resized and grayscaled for model')
plt.imshow(dtrainx[0],cmap='gray')
plt.figure()
plt.title('original image')
plt.imshow(im)
plt.figure()

#plt.imshow(resized_image, cmap='gray')
