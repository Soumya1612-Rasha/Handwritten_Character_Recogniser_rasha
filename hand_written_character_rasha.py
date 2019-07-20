# -*- coding: utf-8 -*-
"""
Created on Fri 19 21:54:21 2019

@author: Soumya Suvra Ghosal
"""
from keras.datasets import mnist
import numpy as np
import os.path as path
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from cnn_model_rasha import CNN_Model_rasha
from Configuration import Cfg
from sklearn.model_selection import train_test_split
np.random.seed(Cfg.random_state)

def load_dataset_rasha():
    '''load dataset'''
	(trainX, trainY),(testX, testY) = mnist.load_data() #Loading the data
	trainX = trainX.reshape((trainX.shape[0], Cfg.image_width, Cfg.image_height, Cfg,channels))     #reshaping the data
	testX = testX.reshape((testX.shape[0], Cfg.image_width, Cfg.image_height, Cfg.channels))  
	trainY = to_categorical(trainY)   #one-hot encoding
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def scale_pixels_rasha(train, test):
    '''function to scale pixels in 0 to 1 range'''
	train_norm = train.astype('float32')   #integer to float
	test_norm = test.astype('float32')
	train_norm /= 255.0  #Scaling the pixels
	test_norm  /= 255.0
	return train_norm, test_norm


def accuracy_rasha(test_x, test_y):
    '''finding accuracy on test data'''
    result = CNN_rasha.predict_data_rasha(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


def plot_history_rasha(histories):
    ''' function to print the learning curves'''
    for i in range(len(histories)):
    		plt.subplot(211)
            title('Cross Entropy Loss')
            plot(histories[i].history['loss'], color='blue', label='train')
            plot(histories[i].history['val_loss'], color='orange', label='test')
            subplot(212)
            title('Classification Accuracy')
            plot(histories[i].history['acc'], color='blue', label='train')
            plot(histories[i].history['val_acc'], color='orange', label='test')
            plt.show()

def main():

    print("\n[info] checking for 'out' folder if exists")
    if not path.exists('out'):
        print("\n[info] out folder not available, creating folder 'out' in the root directory")
        os.mkdir('out')

    print("\n[INFO] loading data from the dataset into variables")
    x_train, y_train, x_test, y_test = load_dataset_rasha()
    x_train, x_test=scale_pixels_rasha(x_train,x_test)
    CNN_rasha=CNN_Model_rasha() #importing class
    if Cfg.dynamic==1:                     #finding best model
         x_test,x_val,y_test,y_val=train_test_split(x_test,y_test,test_size=Cfg.size)
         x_train,x_test,x_val,y_train,y_test,y_val=data()
         CNN_rasha.select_best_model_rasha(data)
    else :
        CNN_rasha.create_model_static_rasha(x_train)    #using static model
        
    [scores,histories]=CNN_rasha.fit_model_rasha(x_train,y_train) #fitting the CNN
    if Cfg.plotHistory:
        plot_history_rasha(histories)        # plot learning curves
    accuracy=accuracy_rasha(x_test,y_test)         #finding accuracy on test data
    print("Accuracy of the model :"+str(accuracy))    


if __name__ == '__main__':
    main()
