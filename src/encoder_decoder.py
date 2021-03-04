import tensorflow as tf
# TODO: might make the encoder not trainable (see example in tf)


def encoder_decoder(output_channels=1):
    inputs = tf.keras.layers.Input(shape=[224, 224, 3])
    rgb = inputs

    # Downsampling, or extracting features
    encoder_stack = encoder()(rgb)
    bottle_neck = encoder_stack[-1]

    # reverse the order of intermediates to get them in the right order
    # for segmenting the frame
    intermediates = reversed(encoder_stack[:-1])
    decoder_stack = decoder()

    # Upsampling, or placing features in the correct position
    x = bottle_neck
    for decode, inter in zip(decoder_stack, intermediates):
        x = decode(x)
        concat_layer = tf.keras.layers.Concatenate()
        x = concat_layer([x, inter])

    # Now we have an approximate depth image, now we need to create the
    # last layer that extracts a final depth estimation
    last_layer = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same')
    x = last_layer(x)

    # Before outputting the result, pass it through a sigmoid activation function
    # so that the output is scaled 0-1
    d_norm = tf.keras.activations.sigmoid(x)
    return tf.keras.Model(inputs=inputs, outputs=d_norm)


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


def upsampler(filters, size):
    # returns an upsampler block that upscales an image
    # conv2dtranspose(strides=2), BN, ReLU
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    return result


def decoder():
    decoder_stack = [
        upsampler(512, 3),  # 7x7  ->  14x14
        upsampler(256, 3),  # 14x14 ->  28x28
        upsampler(128, 3),  # 28x28 ->  56x56
        upsampler(64, 3)  # 56x56 -> 112x112
    ]
    return decoder_stack
