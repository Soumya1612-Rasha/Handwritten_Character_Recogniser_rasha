# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from keras.model import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from sklearn.model_selection import KFold
np.random.seed(Cfg.random_state)
from Configuration import Cfg
import logging
from time import perf_counter
import os.path as path

class CNN_Model_rasha(object):
    def _init_(self):
        self.trained=0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_model_static_rasha(self,x_train):
        model = Sequential()
        model.add(Conv2D(filters= 32, kernel_size=(3, 3),padding="same", activation='relu',kernel_initializer='uniform', input_shape=(Cfg.image_width, Cfg.image_height, Cfg.channels)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model.add(Conv2D(filters=64,kernel_size=(3, 3), activation='relu', kernel_initializer='uniform'))
	    model.add(Conv2D(filters=64,kernel_size=(3, 3), activation='relu', kernel_initializer='uniform'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu',kernel_initializer='uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(Cfg.num_classes, activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])     #compile model
        self.model=model
        self.trained=1
        
    def fit_model_rasha(self, dataX, dataY, n_folds:int=5):
	    scores, histories = list(), list()
	    kfold = KFold(n_folds, shuffle=True, random_state=Cfg.random_state) 	# prepare cross validation
        for train_ix, test_ix in kfold.split(dataX):     	# enumerate splits
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]  # select rows for train and test
		    if self.trained==1 :
                history = self.model.fit(trainX, trainY, epochs=cfg.epochs, batch_size=Cfg.batch_size, validation_data=(testX, testY), verbose=0) # fit model
    		else:
                raise Exception('Model is not trained')
            _, acc = model.evaluate(testX, testY, verbose=0)   # evaluate model
            '> %.3f' % (acc * 100.0))
    		scores.append(acc)   # stores scores
            histories.append(history) #histories scores
        return scores,histories
    
    def predict_data_rasha(self,x_test):
        if self.trained=1:
            return(self.model.predict(x_test))
        else:
            raise Exception('Model is not trained')
    
    def search_best_model_rasha(self,data):
        
        """
        Search the best model using hyperopt & stock best model 
        input data: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from hyperopt import Trials, tpe,rand
        from hyperas import optim
        
        t0 = perf_counter() # timer 
        best_run, best_model,space = optim.minimize(model= create_model_dynamic_rasha, data=data,rseed=Cfg.random_state,algo=tpe.suggest,max_evals=Cfg.evals,eval_space=True,verbose=True,return_space=True)
        self.model = best_model
        self.best_run = best_run
        self.trained=1
        self.logger.info(' best-run:  '+(str(best_run)))
        self.logger.info((' The serach of best model took: ')+ (str(round(perf_counter()-t0,5))) + " " +(".s") )
   
    def create_model_dynamic_rasha(self,x_train,y_train,x_val,y_val):
        
        from hyperas.distributions import choice
        model = Sequential()
        model.add(Conv2D(filters= 32, kernel_size=(3, 3),padding="same", activation='relu',kernel_initializer='uniform', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model.add(Conv2D(filters=64,kernel_size=(3, 3), activation='relu', kernel_initializer='uniform'))
	    model.add(Conv2D(filters=64,kernel_size=(3, 3), activation='relu', kernel_initializer='uniform'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu',kernel_initializer='uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])     #compile model
        model.fit(x_train,y_train,epochs=cfg.epochs,batch_size={{choice=[32,64,128]}},verbose=1,shuffle='False')
        score = model.evaluate(x_val, y_val, verbose=0)
        return {'loss': score, 'status': STATUS_OK, 'model': model}
    
