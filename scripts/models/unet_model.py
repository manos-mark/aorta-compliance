from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow_addons as tfa

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = tfa.layers.GroupNormalization(groups=16, axis=3)(x)
    # x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, 
    #     beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = tfa.layers.GroupNormalization(groups=16, axis=3)(x)
    # x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, 
    #     beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, preprocessing=None, num_filters=64):
    inputs = Input(input_shape)
    
    if type(preprocessing) == keras.Sequential:
        inputs = preprocessing(inputs)

    s1, p1 = encoder_block(inputs, num_filters)
    s2, p2 = encoder_block(p1, 2*num_filters)
    s3, p3 = encoder_block(p2, 4*num_filters)
    s4, p4 = encoder_block(p3, 8*num_filters)

    b1 = conv_block(p4, 16*num_filters)

    d1 = decoder_block(b1, s4, 8*num_filters)
    d2 = decoder_block(d1, s3, 4*num_filters)
    d3 = decoder_block(d2, s2, 2*num_filters)
    d4 = decoder_block(d3, s1, num_filters)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()
