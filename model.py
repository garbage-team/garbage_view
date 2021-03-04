import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers


def adaptive_merge(filters_out, ll_shape, hl_shape):
    low_level = tf.keras.Input(shape=ll_shape)
    high_level = tf.keras.Input(shape=hl_shape)
    inputs = [low_level, high_level]
    # dim_mid = int((low_level_shape[-1] + high_level_shape[-1]) / 16)

    x = tf.keras.layers.Concatenate()([low_level, high_level])
    x = tf.keras.layers.AveragePooling2D(pool_size=ll_shape[1:3])(x)
    x = tf.keras.layers.Conv2D(int(filters_out/8), (1, 1))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters_out, (1, 1))(x)
    x = tf.keras.activations.sigmoid(x)
    out = x * low_level + high_level

    return tf.keras.Model(inputs=inputs, outputs=out)


def dilated_residual(filters_out, shape_in):

    inputs = tf.keras.Input(shape=shape_in)
    x = tf.keras.layers.Conv2D(filters_out, (1, 1,), use_bias=False)(inputs)
    residual = x
    x = tf.keras.layers.Conv2D(filters_out, (3, 3,), padding='same', dilation_rate=2, use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters_out, (3, 3,), padding='same', dilation_rate=2, use_bias=False)(x)
    x += residual
    x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def encoder_decoder(output_channels=1):
    inputs = tf.keras.layers.Input(shape=[224, 224, 3])
    rgb = inputs

    # Downsampling, or extracting features
    encoder_stack = encoder()(rgb)
    bottle_neck = encoder_stack[-1]

    # reverse the order of intermediates to get them in the right order
    # for segmenting the frame
    intermediates = reversed(encoder_stack[:-1])
    ll_shapes = []
    for inter in intermediates:
        ll_shapes.append(inter.shape[1:])  # [h, w, c[
    decoder_stack = decoder(ll_shapes)

    # Bottle neck doing something
    x = tf.keras.layers.Conv2D(512, 1)(bottle_neck)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Upsampling, or placing features in the correct position
    for decode, inter in zip(decoder_stack, intermediates):
        x = decode([x, inter])

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


def upsampler(filters_out, ll_shape, filters_in):
    # returns an upsampler block that upscales an image
    # conv2dtranspose(strides=2), BN, ReLU
    low_level = tf.keras.Input(shape=ll_shape)
    x = dilated_residual(filters_out, ll_shape)(low_level)
    high_level = tf.keras.Input(shape=(ll_shape[0], ll_shape[1], filters_in))
    amb = adaptive_merge(filters_out, x.shape, high_level.shape)
    x = amb([x, high_level])
    x = dilated_residual(filters_out, x.shape)(x)
    x = tf.keras.layers.Conv2DTranspose(filters_out, 3, strides=2,
                                      padding='same',
                                      use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs=[low_level, high_level], outputs=x)


def decoder(ll_shapes):
    decoder_stack = [
        upsampler(512, ll_shapes[0], 512),  # 7x7  ->  14x14
        upsampler(256, ll_shapes[1], 512),  # 14x14 ->  28x28
        upsampler(128, ll_shapes[2], 256),  # 28x28 ->  56x56
        upsampler(64, ll_shapes[3], 128)  # 56x56 -> 112x112
    ]
    return decoder_stack


def normalize(img, depth):
  """Normalizes images: `uint8` -> `float32`."""
  depth = tf.expand_dims(depth, -1)
  IMG_SIZE = 224
  resize = tf.keras.Sequential([
      layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE)
  ])
  img=resize(img)
  depth=resize(depth)
  return tf.cast(img, tf.float32) / 255., tf.cast(depth, tf.float32) / 7000. # 7000? We need to know the maximum value of depth images


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


def saveModel(model, path):

    tf.saved_model.save(model, path)
    print('Model saved to:')
    print(path)
    print('Saving model as tflite model..')
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    tflite_model = converter.convert()
    lite_file = 'D:\wsl\lite_model.tflite'
    open(lite_file, "wb").write(tflite_model)
    print('tflite model saved ')
    return None


def configGPU():
    if tf.config.list_physical_devices('GPU'):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    return None


def loadModel(model_path=False):
    if model_path:
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model successfully!")

    else:
        model = encoder_decoder()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        print("Model generated.")
    model.summary()
    return model


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


if __name__ == '__main__':

    #configGPU()
    path = 'D:/wsl/saved_encoder_decoder'
    model=loadModel() # No argument sets up encoder decoder, enter path to model as argument to use existing
    # Loading the NYU-Dv2 dataset
    #ds = loadNYUDV2()

    #model.fit(dataset, epochs=1)
    #print('Model finished training! Saving..')

    #saveModel(model, path)  # pass a good path
    #testModel(ds,model)

