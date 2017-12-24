#n[16]:


#import cv2
import os
import keras
import tensorflow as tf
from keras.models import Sequential
from keras import regularizers
from keras import backend as K
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, Dropout, MaxPooling2D, Flatten, ZeroPadding2D, Reshape
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.layers.advanced_activations import PReLU, LeakyReLU
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import matplotlib
from keras.utils import np_utils as u
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.datasets import cifar10
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# In[ ]:

# In[23]:

def change_lr(epoch):
    if epoch<50:
        lr = 0.01
    elif epoch<100:
        lr = 0.005
    elif epoch<150:
        lr = 0.001
    else:
        lr = 0.0005
    return lr

def cnn():
    		
    with tf.device('/gpu:1'):
	    model = Sequential()

	   # model.add(BatchNormalization(input_shape = (32,32,3)))

	    #model.add(ZeroPadding2D(padding = ((1,0),(1,0)),input_shape = (32,32,3)))
	    model.add(Conv2D(filters = 32,kernel_size = 3,strides = (1,1),padding = 'same',input_shape = (32,32,3)))
	    model.add(BatchNormalization())
	    #model.add(Activation('relu'))
	    model.add(PReLU(alpha_initializer='zeros'))
	    
	    #model.add(ZeroPadding2D(padding = ((1,0),(1,0))))
	    model.add(Conv2D(filters = 32,kernel_size = 3,strides = (1,1),padding = 'same'))
	    model.add(BatchNormalization())
	    #model.add(Activation('relu'))
	    model.add(PReLU(alpha_initializer='zeros'))
	    
	    model.add(MaxPooling2D((2,2)))
	    model.add(Dropout(0.2))

	    #model.add(ZeroPadding2D(padding = ((1,0),(1,0))))
	    model.add(Conv2D(filters = 64,kernel_size = 3,strides = (1,1),padding = 'same'))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    #model.add(PReLU(alpha_initializer='zeros'))
	    
	    #model.add(ZeroPadding2D(padding = ((1,0),(1,0))))
	    model.add(Conv2D(filters = 64,kernel_size = 3,strides = (1,1),padding = 'same'))
	    model.add(BatchNormalization())
	    #model.add(Activation('relu'))
	    model.add(PReLU(alpha_initializer='zeros'))

	    model.add(MaxPooling2D((2,2)))
	    model.add(Dropout(0.2))
	    
	    #model.add(ZeroPadding2D(padding = ((1,0),(1,0))))
	    model.add(Conv2D(filters = 128,kernel_size = 3,strides = (1,1),padding = 'same'))
	    model.add(BatchNormalization())
	    #model.add(Activation('relu'))
	    model.add(PReLU(alpha_initializer='zeros'))
	    
	    #model.add(ZeroPadding2D(padding = ((1,0),(1,0))))
	    model.add(Conv2D(filters = 128,kernel_size = 3,strides = (1,1),padding = 'same'))
	    model.add(BatchNormalization())
	    #model.add(Activation('relu'))
	    model.add(PReLU(alpha_initializer='zeros'))
	    
	    model.add(MaxPooling2D((2,2)))
	    model.add(Dropout(0.2))
	  
	    model.add(Flatten())

        model.add(Dense(512))
        model.add(BatchNormalization())
        #model.add(Activation('relu'))
        model.add(PReLU(alpha_initializer='zeros'))            

	    model.add(Dropout(0.2))
	    
        model.add(Dense(256))
	    model.add(BatchNormalization())
	    #model.add(Activation('relu'))
	    model.add(PReLU(alpha_initializer='zeros'))
	    model.add(Dropout(0.2))
	   
	    model.add(Dense(10, activation = "softmax"))

	    print(model.summary())

        sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)

	    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	    checkpointer = ModelCheckpoint(filepath = '/others/guest2/cifar10/iit/model.weight.bestprelu.hdf5',verbose =1 , save_best_only=True)

	    lr_change = LearningRateScheduler(change_lr) 
        cb = [checkpointer,lr_change]
        history = model.fit(x_train, y_train, validation_data = (x_val,y_val), batch_size = 64, epochs = 200, callbacks=cb, verbose =1)
	    model.save('/others/guest2/cifar10/iit/model.weight.endprelu.hdf5')
	    print(history.history.keys())
	    # summarize history for accuracy
	    plt.plot(history.history['acc'])
	    plt.plot(history.history['val_acc'])
	    plt.title('model accuracy')
	    plt.ylabel('accuracy')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'test'], loc='upper left')
	    plt.savefig('/others/guest2/cifar10/iit/accuracy_model(prelu).png')
	    # summarize history for loss
	    plt.plot(history.history['loss'])
	    plt.plot(history.history['val_loss'])
	    plt.title('model loss')
	    plt.ylabel('loss')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'test'], loc='upper left')
	    plt.savefig('/others/guest2/cifar10/iit/loss_model(prelu).png')
        scores = model.evaluate(x_test,y_test,batch_size = 32,verbose = 1) 
        print(scores)
	# In[ ]:


        #def cnn_test():
	 #    model = load_model('/others/guest2/iit/model.weight.best(nadam).hdf5')
	  #   y_test = model.predict(x_test, verbose = 0, steps = None)
	   #  print(y_test.shape)
	    # file = open("/others/guest2/result_nadam.txt","w+")
             #for i in len(range(x_test)):
             #file.write(y_test)
	         #file.write(x_test[i] + " " + y_test[i][0] + " " + y_test[i][1] + " " + (y_test[i][2] - y_test[i][0]) + " " + (y_test[i][3] - y_test[i][1]) + "\n")
	#     file.close()


	# In[ ]:

if __name__ == "__main__":
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(x_train.shape,y_train.shape)
    y_train,y_test = u.to_categorical(y_train,10),u.to_categorical(y_test,10)
    print(y_train[0])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
    print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)
    x_train,x_test,x_val = x_train.astype('float32'),x_test.astype('float32'),x_val.astype('float32')
    x_train,x_test,x_val = x_train/255,x_test/255,x_val/255
    cnn()
    #cnn_test()
    print("\n END OF PROGRAM............")
