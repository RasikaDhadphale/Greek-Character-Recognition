#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers

def fit_model(X_train, y_train, X_val, y_val, X_test, y_test):

    model = Sequential()
    
    model.add(Conv2D(8, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (40,40,1)))
    model.add(Conv2D(16, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    
    
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Dense(25, activation = 'softmax'))
    print("Model Summary\n")
    print(model.summary())

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    history = model.fit(
        X_train, 
        y_train, 
        epochs=15, 
        batch_size=64, 
        validation_data=(X_val, y_val)
        # validation_split= 0.1275,
        # shuffle= True
    )

    print("\nPredicticting labels of test data........\n")

    y_pred = model.predict(X_test)
    
    return history, model



