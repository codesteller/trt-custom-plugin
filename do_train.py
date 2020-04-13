from tf_utils.model import Model
from dataset.prepare_dataset import DataBase


# Hyper-parameters
batch_size = 128
epochs = 1
learning_rate = 0.001
input_shape = (150, 150, 3)

# Setup Directories
checkpoint_path = "saved_model/checkpoints/saved_model-{epoch:04d}.h5"
dataset_dir = "/home/codesteller/datasets/kaggle/cv/dogs_cats/train_data"

# Prepare Dataset
db = DataBase(dataset_dir, input_shape=input_shape, batch_size=batch_size)
train_generator, valid_generator = db.data_generators()

# Build Model
cnn_model = Model(input_shape=input_shape,
                  checkpoint_path=checkpoint_path,
                  epochs=epochs,
                  learning_rate=learning_rate,
                  batch_size=batch_size)
cnn_model.build_model()
cnn_model.model.summary()

# Train Model
cnn_model.train_model(train_generator, valid_generator)
cnn_model.prediction("./test_data")

# Save Keras Model to Tensorflow Checkpoint
str_epoch = str(epochs).zfill(4)
final_checkpoint = "saved_model/checkpoints/saved_model-{}.h5".format(str_epoch)
cnn_model.convert_checkpoint(final_checkpoint)
