from tensorflow.keras.layers import Conv2D, AveragePooling2D, Input, Flatten, Dense
from tensorflow.keras.models import Model


def create_base_net(input_shape, latent_dim):
    input = Input(shape=input_shape)
    x = Conv2D(16, (2, 2), activation="tanh", padding="same")(input)
    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
    x = Conv2D(16, (2, 2), activation="tanh", padding="same")(x)
    x = Conv2D(16, (3, 2), activation="tanh", padding="same")(x)
    x = Conv2D(32, (2, 2), activation="tanh", padding="same")(x)
    # x = AveragePooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    x = Dense(latent_dim, activation="tanh")(x)
    model = Model(input, x)
    # model.summary()
    return model
