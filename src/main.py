import tensorflow as tf
import json
import os.path
from src.model import sm_model
from src.image_utils import display_images, resize_normalize, bins_to_depth
from src.loss_functions import wcel_loss, virtual_normal_loss
from src.data_loader import load_nyudv2, load_data, create_dataset, load_tfrecord_dataset


def main():
    config_gpu()
    model = sm_model()
    model = optimize_compile_model(model)
    model.summary()

    checkpoint_filepath = '../tmp/model_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True)

    ds_train = load_tfrecord_dataset("../data/garbage_record_train")
    ds_val = load_tfrecord_dataset("../data/garbage_record_validation", augment=False)
    history = model.fit(ds_train, epochs=1, validation_data=ds_val,
              callbacks=[model_checkpoint_callback, tf.keras.callbacks.TerminateOnNaN()])
    save_model(model, history, path='../models/augmented')
    for rgb, d in ds_val.take(1):
        test_model(rgb[0], d[0], model)
    return None

# TODO Clean this code up, refactor functions to appropriate files
def custom_loss(gt, pred):
    loss_wcel = wcel_loss(gt, pred)
    loss_vnl = 6 * virtual_normal_loss(gt, pred)
    return loss_wcel + loss_vnl


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
        #tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
            #tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    return None


def test_model(rgb, d, model):
    # Takes a rgb image with corresponding ground truth and the model to test
    print("Testing model...")
    rgb = tf.expand_dims(rgb, 0)  # Convert from [h, w, c] to [1, h, w, c]
    d_est = model.predict(rgb)
    d_est = bins_to_depth(d_est)
    display_images([rgb[0], d, d_est[0]])
    return None


def save_model(model, history, path):
    """
    Saves the model input to the path input
    :param model: The model object to be saved
    :param history: The history generated from training
    :param path: The path to where the model and history should be saved.
    :return:
    """
    tf.saved_model.save(model, path)
    json_file = path + ".json"
    if os.path.exists(json_file):
        with open(json_file, 'r+') as f:
            data = json.load(f)
            for key in data:
                values = history.history[key]
                for value in values:
                    data[key].append(value)
            f.seek(0)
            json.dump(data, f)
    else:
        with open(json_file, 'w') as f:
            json.dump(history.history, f)
    print('Model saved to:', path)
    return None


def load_model(model_path='../models/model'):
    print("Loading model: "+ model_path)
    model = tf.keras.models.load_model(model_path, compile=False) # TODO Add the correct settings to the optimizer
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
