import tensorflow as tf
import tensorflow_addons as tfa


def adaptive_merge(ll_filters_in, hl_filters_in, filters_out):
    # An adaptive merge block for merging low level and high level features
    # Must know the number of filters in the inputs and the number of filters out
    # returns a model with inputs as [low_level, high_level] and outputs [merged]

    # Defining the inputs of the model
    low_level = tf.keras.Input(shape=(None, None, ll_filters_in))
    high_level = tf.keras.Input(shape=(None, None, hl_filters_in))
    inputs = [low_level, high_level]

    # Calculating the output of the model
    x = tf.keras.layers.Concatenate()([low_level, high_level])
    x = tfa.layers.AdaptiveAveragePooling2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(int(filters_out/8), (1, 1))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters_out, (1, 1))(x)
    x = tf.keras.activations.sigmoid(x)
    out = x * low_level + high_level

    return tf.keras.Model(inputs=inputs, outputs=out)


def dilated_residual(filters_in, filters_out):
    # A dilated residual block
    # returns a model that takes an input and returns an output
    inputs = tf.keras.Input(shape=(None, None, filters_in))

    x = tf.keras.layers.Conv2D(filters_out, (1, 1), use_bias=False)(inputs)
    residual = x
    x = tf.keras.layers.Conv2D(filters_out, (3, 3),
                               padding='same', dilation_rate=2, use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters_out, (3, 3),
                               padding='same', dilation_rate=2, use_bias=False)(x)
    x += residual
    out = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=out)


def prediction_layer(filters_in, depth_bins):
    # The prediction layer of the model
    # Takes any amount of filters as an input and applies a convolution,
    # and then a softmax in the channels dimension, returns the depth bins
    inputs = tf.keras.Input(shape=(None, None, filters_in))
    x = tf.keras.layers.SpatialDropout2D(0.1)(inputs)
    x = tf.keras.layers.Conv2D(depth_bins, 3, strides=1,
                               padding='same', use_bias=True)(x)
    x_softmax = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs=inputs, outputs=[x, x_softmax])


def decode_layer(ll_filters_in, hl_filters_in, filters_out):
    ll_in = tf.keras.Input(shape=(None, None, ll_filters_in))
    hl_in = tf.keras.Input(shape=(None, None, hl_filters_in))
    x = adaptive_merge(ll_filters_in, hl_filters_in, hl_filters_in)((ll_in, hl_in))
    x = dilated_residual(hl_filters_in, hl_filters_in)(x)
    x = tf.keras.layers.Conv2DTranspose(filters_out, 3, strides=2,
                                        padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs=[ll_in, hl_in], outputs=out)
