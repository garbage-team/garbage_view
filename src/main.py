# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from encoder_decoder import encoder_decoder
from model_trainer import train_model_to_data
from src.image_utils import bins_to_depth
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def main():
    configGPU()
    model = trash_view_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    nyu = loadNYUDV2(batch=8, shuffle=False)
    model.fit(nyu, epochs=3)
    print('Model finished training!')
    path= 'D:/wsl/modelv2'
    saveModel(model, path)  # pass a good path
    testModel(nyu,model)
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


def normalize(img, depth):
    """Normalizes images: `uint8` -> `float32`."""
    depth = tf.expand_dims(depth, -1)
    img_size = 224
    resize = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)
    ])
    img = resize(img)
    depth = resize(depth)
    return tf.cast(img, tf.float32) / 255., tf.cast(depth, tf.float32)


def loadNYUDV2(batch=32,shuffle=True):
    nyudv2, info = tfds.load('nyu_depth_v2', split='train', with_info=True, shuffle_files=True, as_supervised=True,
                             data_dir='D:/wsl/tensorflow_datasets')
    nyudv2 = nyudv2.map(
        normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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


def testModel(ds,model):
    print("Testing model...")
    for (image,label) in ds.take(1):
        rgb=image.numpy()
        d=label.numpy()
    rgb = tf.expand_dims(rgb, 0)
    d_est = model.predict(rgb)
    plt.subplot(1, 3, 1)
    plt.imshow(rgb[0])
    plt.subplot(1, 3, 2)
    plt.imshow(d)
    plt.subplot(1, 3, 3)
    plt.imshow(d_est[0])
    plt.show()

    return None


def saveModel(model, path):

    tf.saved_model.save(model, path)
    print('Model saved to:')
    print(path)

    return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
