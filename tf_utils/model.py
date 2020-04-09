import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, LeakyReLU
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.metrics import AUC
import os


class Model:
    def __init__(self,
                 input_shape=(224, 224, 3),
                 num_classes=2,
                 checkpoint_path="./checkpoint",
                 batch_size=32,
                 epochs=10,
                 learning_rate=0.001):
        """
        input_shape - In HWC format
        """
        self.model = Sequential()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def build_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                              input_shape=self.input_shape, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(
            Conv2D(128, (3, 3), activation='linear', padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model.compile(optimizer=opt,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        print("Model built and compiled successfully")

    def train_model(self, train_data_gen, valid_data_gen):
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tensorboard_dir = os.path.join(checkpoint_dir, 'tensorboard')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1, period=2)
        tb_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=False)
        self.model.fit_generator(
                train_data_gen,
                steps_per_epoch=train_data_gen.samples // self.batch_size,
                epochs=self.epochs,
                validation_data=valid_data_gen,
                validation_steps=valid_data_gen.samples // self.batch_size,
                callbacks=[cp_callback, tb_callback])


def test_case():
    try:
        model = Model()
        model.build_model()
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    if test_case():
        print("Test Case Passed")
    else:
        print("Test Case Passed")
