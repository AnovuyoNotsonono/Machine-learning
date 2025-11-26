#!/usr/bin/env python

# Importing the libraries to use
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Defining the model

def define_model():
    model =Sequential([Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(50,50,3)),
                  MaxPooling2D(pool_size=(2,2)),

                  Conv2D(64,(3,3), activation='relu',padding='same'),
                  MaxPooling2D(pool_size=(2,2)),

                 Conv2D(128,(3,3), activation='relu',padding='same'),
                 MaxPooling2D(pool_size=(2,2)),

                 Flatten() ,
                 Dense(256, activation='relu') ,

                 Dropout(0.2),

                 Dense(1, activation='sigmoid')]
                        )
    # Compile and return the model
    return model.compile(optimizer='adamW',metrics=['accuracy'],loss='binary_crossentropy',)

