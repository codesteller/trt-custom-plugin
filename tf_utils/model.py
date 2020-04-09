import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, LeakyReLU
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.backend import get_session
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
        self.model.add(InputLayer(input_shape=self.input_shape))
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
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1, period=1)
        # cp_callback = ModelCheckpoint(filepath=self.checkpoint_path, monitor='val_acc', verbose=1,
        #                               save_best_only=True, mode='max')
        tb_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=False)
        self.model.fit_generator(
            train_data_gen,
            steps_per_epoch=train_data_gen.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=valid_data_gen,
            validation_steps=valid_data_gen.samples // self.batch_size,
            callbacks=[cp_callback, tb_callback])

    def convert_checkpoint(self, final_checkpoint):
        checkpoint_dir = os.path.dirname(final_checkpoint)
        basename = os.path.basename(final_checkpoint).split(".")[0]
        save_path = os.path.join(checkpoint_dir, "tf_ckpt", "final_model.ckpt")
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        self.model.load_weights(final_checkpoint)
        sess = get_session()
        saver.save(sess, save_path)

    def save(self, frozen_filename):
        # First freeze the graph and remove training nodes.
        output_names = self.model.output.op.name
        sess = get_session()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
        # Save the model
        with open(frozen_filename, "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())


def test_case():
    try:
        model = Model()
        model.build_model()
        return True
    except Exception as e:
        print(e)
        return False


def test_case2():
    final_checkpoint = "/home/codesteller/workspace/ml_workspace/trt-custom-plugin/saved_model/" \
                       "checkpoints/saved_model-0001.h5"
    model = Model(input_shape=(150, 150, 3))
    model.build_model()
    model.convert_checkpoint(final_checkpoint)
    print("done")


if __name__ == "__main__":
    if test_case():
        print("Test Case Passed")
    else:
        print("Test Case Passed")

    test_case2()

