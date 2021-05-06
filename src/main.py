import tensorflow as tf
import json
import os.path
from src.model import sm_model
from src.image_utils import display_images, resize_normalize, bins_to_depth
from src.loss_functions import wcel_loss, virtual_normal_loss
from src.fill_rate_loss import actual_fill_rate_loss
from src.data_loader import load_nyudv2, load_data, create_dataset, load_tfrecord_dataset


def main():
    # config_gpu()
    path = 'D:/github/garbage_view/models/model_small'
    ds_path = 'D:/tensorflow_datasets'
    # model = load_model(path)
    model = sm_model()
    model.summary()
    model = optimize_compile_model(model, fov="intel")

    training_loop(model)
    ds = load_tfrecord_dataset("D:/garbage_record_validation", batch=1, augment=False)
    for rgb, d in ds.take(2):
        test_model(rgb[0], d[0], model)
    return None


def training_loop(model):
    path = 'D:/github/garbage_view/models/model_small_aspp'
    ds_path = 'D:/tensorflow_datasets'
    epochs = [10]
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0001,
        end_learning_rate=0.00001,
        decay_steps=200000,
        power=0.9
    )
    for i in range(len(epochs)):
        optimize_compile_model(model, fov="kinect", lr=lr)
        # ds_train = load_tfrecord_dataset("D:/garbage_record_train", batch=1)
        # ds_val = load_tfrecord_dataset("D:/garbage_record_validation", batch=1, augment=False)
        ds_train, _ = load_nyudv2(batch=4, shuffle=True, ds_path=ds_path, split='train')
        ds_val, _ = load_nyudv2(batch=4, shuffle=True, ds_path=ds_path, split='validation')
        history = model.fit(ds_train, epochs=epochs[i], validation_data=ds_val,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        save_model(model, history, path=path)


# TODO Clean this code up, refactor functions to appropriate files
def custom_loss(gt, pred, fov="kinect"):
    loss_wcel = wcel_loss(gt, pred)
    loss_vnl = 6 * virtual_normal_loss(gt, pred, fov=fov)
    # loss_frl = 0.1 * actual_fill_rate_loss(gt, pred, fov=fov)
    return loss_wcel + loss_vnl # + loss_frl


def custom_accuracy(gt, pred):
    pred_depth = bins_to_depth(pred)
    return 1. / (1. + tf.keras.metrics.MSE(gt, tf.expand_dims(pred_depth, axis=-1)))


def save_to_tflite(model_path):
    # Convert the saved tf2 model as a tflite object
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    open(model_path + "_lite_model.tflite", "wb").write(tflite_model)
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
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    # model.compile(optimizer=optimizer,
    #               loss=wcel_loss,
    #               metrics=['accuracy'])
    print("Loaded existing model successfully!")
    model.summary()
    model = optimize_compile_model(model, fov="kinect")
    return model


def optimize_compile_model(model, fov="kinect", lr=0.0005):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=lambda x, y: custom_loss(x, y, fov=fov),
                  metrics=[custom_accuracy])
    return model


if __name__ == '__main__':
    main()
