from os import listdir
from PIL import Image
import os
from plot_CTU import CTU_Partition
import numpy as np

def get_all_yuv_files(dataset_path) -> list[str]:
    # logger.info(f'Get dataset path {self.dataset_path}')
    only_files = [f for f in listdir(dataset_path) if f.endswith('.yuv')]
    only_files.sort()
    # logger.debug(f"All files in dataset path: {only_files}")
    return only_files

def get_yuv_image_width_hight(origin_path: str, filename: str):
        origin_image = Image.open(os.path.join(origin_path, filename))
        size = origin_image.size
        print(f'file_name: {filename}, size: {size}')
        width = size[0]
        height = size[1]
        return (width, height)

path = '/home/woody/dataset/training_data/DIV2K_QP32_nDBF'

origin_path = '/home/woody/dataset/DIV2K_train_HR'
files = get_all_yuv_files(path)
for file in files:
    name = file.split('.')[0]
    name = name.split('_')[0]
    file_name = f'{name}.png'
    size = get_yuv_image_width_hight(origin_path, file_name)
    img = os.path.join(path, file)
    partition = CTU_Partition(img, size)
    luma_map = partition.get_partition_map('y')
    chroma_map = partition.get_partition_map('u')
    luma_title = f'{name}_luma_CTU.png'
    luma = os.path.join(path,luma_title)
    partition.save_img(luma, luma_map)
    print(f'Save img {luma}')
    chroma_title = f'{name}_chroma_CTU.png'
    chroma = os.path.join(path,chroma_title)
    partition.save_img(chroma, chroma_map)
    print(f'Save img {chroma}')
# print(files)