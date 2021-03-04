# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from encoder_decoder import encoder_decoder
from model_trainer import train_model_to_data
from src.image_utils import bins_to_depth


def main():
    model = trash_view_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return None


def trash_view_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    enc_dec = encoder_decoder()
    [x, x_softmax] = enc_dec(inputs)
    depth = bins_to_depth(x_softmax)
    return tf.keras.Model(inputs=inputs, outputs=depth)


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
