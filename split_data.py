import os
import numpy as np
import shutil
import os.path

img_file = r"img"
root_dir = r"data"
valid_ratio = 0.2

if not os.path.exists(root_dir + '/train'):
    os.makedirs(root_dir + '/train')
if not os.path.exists(root_dir + '/valid'):
    os.makedirs(root_dir + '/valid')

all_filename = os.listdir(img_file)
np.random.shuffle(all_filename)
train_files, valid_files = np.split(np.array(all_filename), [int(len(all_filename) * (1 - valid_ratio))])

print("*****************************")
print('Total images: ', len(all_filename))
print('Training: ', len(train_files))
print('Testing: ', len(valid_files))
print("*****************************")


for name in train_files:
    src = img_file +"/"+ name
    dst = root_dir + '/train/nofight/' + name
    os.rename(src,dst)

for name in valid_files:
    src = img_file +"/"+ name
    dst = root_dir + '/valid/nofight/' + name
    os.rename(src,dst)
