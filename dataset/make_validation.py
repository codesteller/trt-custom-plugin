from glob import glob
import os
import shutil


dog_path = "/home/codesteller/datasets/kaggle/cv/dogs_cats/train/train/dogs"
cat_path = "/home/codesteller/datasets/kaggle/cv/dogs_cats/train/train/cats"

dogs = glob(os.path.join(dog_path, "*.jpg"))
cats = glob(os.path.join(cat_path, "*.jpg"))
print(len(dogs))
print(len(cats))

from random import shuffle
shuffle(dogs)
shuffle(cats)

dogs_valid = dogs[:2500]
cats_valid = cats[:2500]

# Copy to valid folder dogs
cp_path = "/home/codesteller/datasets/kaggle/cv/dogs_cats/valid/dogs"
for ifile in dogs_valid:
    shutil.move(ifile, cp_path)

# Copy to valid folder dogs
cp_path = "/home/codesteller/datasets/kaggle/cv/dogs_cats/valid/cats"
for ifile in cats_valid:
    shutil.move(ifile, cp_path)