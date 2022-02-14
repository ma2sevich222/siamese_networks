from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D,Input, Flatten, Dense, Dropout, Lambda





def create_base_net(input_shape):

    input = Input(shape=input_shape)
    x = Conv2D(16, (2, 2), activation="tanh", padding="same")(input)
    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
    x = Conv2D(16, (2, 2), activation="tanh", padding="same")(x)
    x = Conv2D(16, (3, 2), activation="tanh", padding="same")(x)
    x = Conv2D(32, (2, 2), activation="tanh", padding="same")(x)
    # x = AveragePooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    x = Dense(10, activation="tanh")(x)
    model = Model(input, x)
    model.summary()
    return model