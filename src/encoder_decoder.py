import tensorflow as tf
from model_blocks import adaptive_merge, dilated_residual, prediction_layer, decode_layer


def encoder_decoder(output_channels=1):
    inputs = tf.keras.layers.Input(shape=[224, 224, 3])
    rgb = inputs

    # Downsampling, or extracting features
    encoder_stack = encoder()(rgb)
    bottle_neck = encoder_stack[-1]

    # reverse the order of intermediates to get them in the right order
    # for segmenting the frame
    intermediates = [item for item in reversed(encoder_stack[:-1])]
    intermediate_filters = [item for item in reversed([96, 144, 192, 576])]
    decode_filters = [512, 512, 256, 128, 128]

    # Upsampling, and placing features in the correct position
    x = bottle_neck  # Here we might apply ASPP
    x = tf.keras.layers.Conv2DTranspose(512, 3, strides=2,
                                        padding='same')(x)
    for i in range(len(decode_filters) - 1):
        u = dilated_residual(intermediate_filters[i], decode_filters[i])(intermediates[i])
        x = decode_layer(decode_filters[i], decode_filters[i], decode_filters[i+1])((u, x))

    # Now we add the prediction layer to the model
    x, x_softmax = prediction_layer(decode_filters[-1], 150)(x)
    return tf.keras.Model(inputs=inputs, outputs=[x, x_softmax])


def encoder():
    base_encoder = tf.keras.applications.MobileNetV2(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet'
    )

    # Finding the layers where we want to extract the intermediate results
    layer_names = [
            "block_1_expand_relu",   # 112x112
            "block_3_expand_relu",   # 56x56
            "block_6_expand_relu",   # 28x28
            "block_13_expand_relu",  # 14x14
            "block_16_project",      # 7x7
    ]
    layers = [base_encoder.get_layer(name).output for name in layer_names]

    # Create the final encoder down stack
    encoder_stack = tf.keras.Model(inputs=base_encoder.input, outputs=layers)
    return encoder_stack
