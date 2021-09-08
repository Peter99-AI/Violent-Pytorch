import os
import numpy as np
import shutil
import os.path


def split_data(path_train, path_valid):
    img_file = r"img"
    root_dir = r"data"
    valid_ratio = 0.2

    all_filename = os.listdir(img_file)
    np.random.shuffle(all_filename)
    train_files, valid_files = np.split(np.array(all_filename), [int(len(all_filename) * (1 - valid_ratio))])

    n1 = len(train_files) % 15
    train_files = train_files[n1:]
    m1 = len(valid_files) % 15
    valid_files = valid_files[m1:]

    print("*****************************")
    print('Total images: ', len(all_filename))
    print('Training: ', len(train_files))
    print('Testing: ', len(valid_files))
    print("*****************************")



    for name in train_files:
        src = img_file +"/"+ name
        dst = root_dir + path_train + name
        os.rename(src,dst)

    for name in valid_files:
        src = img_file +"/"+ name
        dst = root_dir + path_valid + name
        os.rename(src,dst)
