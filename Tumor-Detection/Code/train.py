#!/usr/bin/env python


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import define_model
import pickle


def get_training_history():
    """Optional function to load previously saved history inside another function"""
    with open("history.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":

    # Specifying the rules and parameters

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True,
                                  vertical_flip=True,shear_range=0.2,zoom_range=0.2,
                                  width_shift_range=0.2, height_shift_range=0.2,)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_datagenerator = train_datagen.flow_from_directory('../data/train/',target_size=(50,50), batch_size=32,
                                                       class_mode='binary')

    test_datagenerator = test_datagen.flow_from_directory('../data/test/',target_size=(50,50), batch_size=32,
                                                      class_mode='binary')

    # Apply EarlyStopping regularization to stop model from ovefitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True)

    # Call the model
    model = define_model()

    #Train the model
    history = model.fit(train_datagenerator, epochs=100, validation_data=test_datagenerator, callbacks=[early_stop])
   #Save the model
    model.save("../model/Tumor_detector.keras")

  # Save history object to a file so it can be imported later

    with open("history.pkl", "wb") as f:
        pickle.dump(history.history, f)
