import os
from keras.preprocessing.image import ImageDataGenerator


class DataBase:
    def __init__(self, dbpath, input_shape=(224, 224, 3), batch_size=64):
        """

        :param dbpath:  Path to the Dataset. DataBase works on a definite folder structure.
                        Override the functions "fetch_data_paths" and "data_generators" for custom dataset.
        :param input_shape: In format HWC
        """
        self.dbpath = dbpath
        self.input_shape = input_shape
        self.train_dir = os.path.join(self.dbpath, 'train')
        self.valid_dir = os.path.join(self.dbpath, 'valid')
        self.im_height = input_shape[0]
        self.im_width = input_shape[1]
        self.channels = input_shape[2]
        self.batch_size = batch_size

        self.fetch_data_paths()

    def fetch_data_paths(self):
        train_cats_dir = os.path.join(self.train_dir, 'cats')  # directory with our training cat pictures
        train_dogs_dir = os.path.join(self.train_dir, 'dogs')  # directory with our training dog pictures
        validation_cats_dir = os.path.join(self.valid_dir, 'cats')  # directory with our validation cat pictures
        validation_dogs_dir = os.path.join(self.valid_dir, 'dogs')  # directory with our validation dog pictures

        num_cats_tr = len(os.listdir(train_cats_dir))
        num_dogs_tr = len(os.listdir(train_dogs_dir))

        num_cats_val = len(os.listdir(validation_cats_dir))
        num_dogs_val = len(os.listdir(validation_dogs_dir))

        total_train = num_cats_tr + num_dogs_tr
        total_val = num_cats_val + num_dogs_val

        print('total training cat images:', num_cats_tr)
        print('total training dog images:', num_dogs_tr)

        print('total validation cat images:', num_cats_val)
        print('total validation dog images:', num_dogs_val)
        print("--")
        print("Total training images:", total_train)
        print("Total validation images:", total_val)

    def data_generators(self):
        # Prepare Training Data
        train_image_generator = ImageDataGenerator(rotation_range=40,
                                                   width_shift_range=0.2,
                                                   height_shift_range=0.2,
                                                   shear_range=0.2,
                                                   zoom_range=0.2,
                                                   channel_shift_range=10,
                                                   horizontal_flip=True,
                                                   fill_mode='nearest',
                                                   rescale=1. / 255)
        train_batches = train_image_generator.flow_from_directory(self.train_dir,
                                                                  target_size=(self.im_height, self.im_width),
                                                                  interpolation='bicubic',
                                                                  class_mode='categorical',
                                                                  shuffle=True,
                                                                  batch_size=self.batch_size)

        # Prepare Validation Data
        valid_image_generator = ImageDataGenerator(rescale=1. / 255)
        valid_batches = valid_image_generator.flow_from_directory(self.train_dir,
                                                                  target_size=(self.im_height, self.im_width),
                                                                  interpolation='bicubic',
                                                                  class_mode='categorical',
                                                                  shuffle=True,
                                                                  batch_size=self.batch_size)

        return train_batches, valid_batches


def test_case1():
    db = DataBase("/home/codesteller/datasets/kaggle/cv/dogs_cats/train_data")
    dataset_generators = db.data_generators()
    return dataset_generators


def test_case2(data_generators):
    import matplotlib.pyplot as plt
    train_data_gen = data_generators[0]
    # Check Samples of Training Data
    sample_training_images, sample_training_labels = next(train_data_gen)

    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    plotImages(sample_training_images[:5])
    print(sample_training_labels[:5])


if __name__ == '__main__':
    dataset_generators = test_case1()

    if dataset_generators:
        print("Test Case - 1:  PASSED")
    else:
        print("Test Case - 1:  FAILED")

    try:
        test_case2(dataset_generators)
        print("Test Case - 2:  PASSED")
    except Exception as e:
        print("Test Case - 2:  FAILED \n{}".format(e))
