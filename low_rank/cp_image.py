import os
import random
import shutil

target_path = "/mnt/ramdisk/ImageNet/fewshot4_train_s10/"
source_path = "/mnt/ramdisk/ImageNet/fewshot4_train/"


if not os.path.exists(target_path):
    os.mkdir(target_path)

img_floder_files  = os.listdir(source_path)
random.shuffle(img_floder_files)
# for img_floder_file in img_floder_files[:200]:
for img_floder_file in img_floder_files[:]:
    if not os.path.exists(target_path + img_floder_file):
        os.mkdir(target_path + img_floder_file)
    img_floder_path = os.path.join(source_path, img_floder_file)
    imgs = os.listdir(img_floder_path)
    random.shuffle(imgs)
    l = 2
    for i in range(l):
        shutil.copy(img_floder_path + "/" + imgs[i],  target_path + img_floder_file + "/" + imgs[i] )