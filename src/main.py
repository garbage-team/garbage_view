# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import init_model
from src.image_utils import display_images, resize_normalize
from src.loss_functions import custom_loss
from src.data_loader import load_data


def main():
    # configGPU()
    model = depth_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss=custom_loss,
                  metrics=['accuracy'])
    #nyu = loadNYUDV2(batch=8, shuffle=False)
    #model.fit(nyu, epochs=3)
    #print('Model finished training!')
    #path = 'D:/wsl/modelv2'
    #save_model(model, path)  # pass a good path
    #test_model([(rgb, d) for (rgb, d) in nyu.take(1)], model)
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


def loadNYUDV2(batch=32,shuffle=True):
    nyudv2, info = tfds.load('nyu_depth_v2', split='train', with_info=True, shuffle_files=True, as_supervised=True,
                             data_dir='D:/wsl/tensorflow_datasets')
    nyudv2 = nyudv2.map(
        resize_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    nyudv2 = nyudv2.cache()
    BATCH_SIZE = batch

    if batch and shuffle:
        SHUFFLE_SIZE = int(5 * 47584 / BATCH_SIZE) # Replace number with a num_elements type of value
        nyudv2 = nyudv2.shuffle(SHUFFLE_SIZE)  # check that this is the correct way of calling num_examples
        nyudv2 = nyudv2.batch(BATCH_SIZE)
    if batch and not shuffle:
        nyudv2 = nyudv2.batch(BATCH_SIZE)
    if shuffle and not batch:
        SHUFFLE_SIZE = 5*47584  # Replace number with a num_elements type of value
        nyudv2 = nyudv2.shuffle(SHUFFLE_SIZE)  # check that this is the correct way of calling num_examples

    nyudv2 = nyudv2.prefetch(tf.data.experimental.AUTOTUNE)

    return nyudv2


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
    display_images([rgb[0], d, d_est[0]])
    return None


def save_model(model, path):
    tf.saved_model.save(model, path)
    print('Model saved to:', path)
    return None


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded existing model successfully!")
    model.summary()
    return model


if __name__ == '__main__':
    main()
