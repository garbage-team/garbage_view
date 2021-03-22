import tensorflow as tf
from src.image_utils import bins_to_depth
from src.model_blocks import *
from src.config import cfg


def encoder_decoder():
    input_size = list(cfg["input_size"])
    inputs = tf.keras.layers.Input(shape=input_size)
    rgb = inputs

    # Downsampling, or extracting features
    encoder_model = encoder()
    for layer in encoder_model.layers:
        layer.trainable = False
    encoder_stack = encoder_model(rgb)
    bottle_neck = encoder_stack[-1]

    # reverse the order of intermediates to get them in the right order
    # for segmenting the frame
    intermediates = [item for item in reversed(encoder_stack[:-1])]
    intermediate_filters = [item for item in reversed(cfg["encoder_filters"])]
    decode_filters = cfg["decoder_filters"]

    # Upsampling, and placing features in the correct position
    x = bottle_neck  # Here we might apply ASPP
    x = tf.keras.layers.Conv2DTranspose(cfg["model_bottleneck_channels"], 3, strides=2,
                                        padding='same')(x)
    for i in range(len(decode_filters) - 1):
        u = dilated_residual(intermediate_filters[i], decode_filters[i])(intermediates[i])
        x = decode_layer(decode_filters[i], decode_filters[i], decode_filters[i+1])((u, x))

    # Now we add the prediction layer to the model
    x, x_softmax = prediction_layer(decode_filters[-1], cfg["depth_bins"])(x)
    return tf.keras.Model(inputs=inputs, outputs=[x, x_softmax])


def depth_model(shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=shape)
    [x, x_softmax] = full_model(shape)(inputs)
    depth = bins_to_depth(x_softmax)
    return tf.keras.Model(inputs=inputs, outputs=depth)


def sm_model(shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=shape)
    [x, x_softmax] = full_model(shape)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x_softmax)


def full_model(shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=shape)
    [x, x_softmax] = encoder_decoder()(inputs)
    return tf.keras.Model(inputs=inputs, outputs=[x, x_softmax])
