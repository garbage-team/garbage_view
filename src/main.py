# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from encoder_decoder import encoder_decoder
from model_trainer import train_model_to_data


def main():
    model = encoder_decoder()
    model.summary()
    return None


def save_to_tflite(model):
    path = "../model/"

    # Save the model as a tf SavedModel object
    tf.saved_model.save(model, path)

    # Convert the saved tf2 model as a tflite object
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()
    open("../model/lite_model.tflite", "wb").write(tflite_model)
    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
