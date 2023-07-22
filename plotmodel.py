from keras.models import Sequential
from keras.layers import Dense,Conv2D,BatchNormalization,MaxPool2D,Dropout,Flatten
from keras.utils import plot_model

model = Sequential()

def model_arch():
    model.add(
        Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            activation="relu",
            input_shape=(100,100,3),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=384,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())  # [+] Convert the Conv2D objects into one List.

    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    # [+] 7th Dense layer
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    # [+] 8th output layer
    model.add(Dense(8, activation="softmax"))
    # model.summary()

model_arch()
# Plot the model architecture
plot_model(model, to_file='model.png', show_shapes=True)
print("done plotting model")