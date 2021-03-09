# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import sm_model
from src.image_utils import display_images, resize_normalize, bins_to_depth
from src.loss_functions import wcel_loss
from src.data_loader import load_nyudv2, load_data


def main():
    configGPU()
    path = 'D:/wsl/modelv2_1_wcel'
    model = load_model(path)
    ds = load_nyudv2(shuffle=False, batch=4)
    model.fit(ds, epochs=4)

    save_model(model, path)
    img_paths=[('D:/wsl/17_Color.png', 'D:/wsl/17_Depth.raw')]
    [(rgb, d)] = load_data(img_paths)
    rgb ,d = resize_normalize(rgb, d, max_depth=80000)
    test_model(rgb, d, model)
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





def configGPU():
    if tf.config.list_physical_devices('GPU'):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    return None


def test_model(rgb, d, model):
    # Takes a list of [(rgb, d)] in rgb_d
    print("Testing model...")
    rgb = tf.expand_dims(rgb, 0)  # Convert from [h, w, c] to [1, h, w, c]
    d_est = model.predict(rgb)
    d_est = bins_to_depth(d_est)
    display_images([rgb[0], d, d_est[0]])
    return None


def save_model(model, path):
    tf.saved_model.save(model, path)
    print('Model saved to:', path)
    return None


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False) # TODO Add the compilation arguments here instead
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                  loss=wcel_loss,
                  metrics=['accuracy'])
    print("Loaded existing model successfully!")
    model.summary()
    return model


if __name__ == '__main__':
    main()
