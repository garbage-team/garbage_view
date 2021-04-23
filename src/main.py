import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import sm_model
from src.image_utils import display_images, resize_normalize, bins_to_depth
from src.loss_functions import wcel_loss, virtual_normal_loss
from src.data_loader import load_nyudv2, load_data, create_dataset


def main():
    config_gpu()
    path = 'D:/garbage_view/model'
    ds_path = 'D:/tensorflow_datasets'
    model = load_model(path)
    # model = sm_model()
    model.summary()

    checkpoint_filepath = '../tmp/model_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    ds = load_nyudv2(shuffle=True, batch=4)
    # ds = create_dataset()
    model.fit(ds, epochs=1, callbacks=[model_checkpoint_callback, tf.keras.callbacks.TerminateOnNaN()])
    save_model(model, path='../models/model')
    img_paths = [('D:/wsl/17_Color.png', 'D:/wsl/17_Depth.raw')]
    # [(rgb, d)] = load_data(img_paths)
    # rgb, d = resize_normalize(rgb, d, max_depth=80000)
    # test_model(rgb, d, model)
    return None


def custom_loss(gt, pred):
    loss_wcel = wcel_loss(gt, pred)
    loss_vnl = 6 * virtual_normal_loss(gt, pred)
    return loss_wcel + loss_vnl


def custom_accuracy(gt, pred):
    return 1


def save_to_tflite(model):
    path = "../model/"

    # Save the model as a tf SavedModel object
    tf.saved_model.save(model, path)

    # Convert the saved tf2 model as a tflite object
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()
    open("../model/lite_model.tflite", "wb").write(tflite_model)
    return None


def config_gpu():
    # Configures GPU memory to avoid some common issues with cuda
    if tf.config.list_physical_devices('GPU'):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    return None


def test_model(rgb, d, model):
    # Takes a rgb image with corresponding ground truth and the model to test
    print("Testing model...")
    rgb = tf.expand_dims(rgb, 0)  # Convert from [h, w, c] to [1, h, w, c]
    d_est = model.predict(rgb)
    d_est = bins_to_depth(d_est)
    display_images([rgb[0], d, d_est[0]])
    return None


def save_model(model, path):
    """
    Saves the model input to the path input
    :param model: 
    :param path:
    :return:
    """
    tf.saved_model.save(model, path)
    print('Model saved to:', path)
    return None


def load_model(model_path='../models/model'):
    model = tf.keras.models.load_model(model_path, compile=False) # TODO Add the correct settings to the optimizer
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    # model.compile(optimizer=optimizer,
    #               loss=wcel_loss,
    #               metrics=['accuracy'])
    print("Loaded existing model successfully!")
    model.summary()
    model = optimize_compile_model(model)
    return model


def optimize_compile_model(model):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=custom_loss,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
