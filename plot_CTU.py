import cv2
import numpy as np
import math
import yuvio
import os
from PIL import Image

class CTU_Partition:
    def __init__(self, file_path: str, size: tuple, dist: str = './'):
        """
        file_path: yuv file path,
        ex: '/home/woody/dataset/training_data/DIV2K_QP32_nDBF/0001_reco.yuv'
        size: yuv image size (height, width)
        ex: (2040, 1404)
        """
        self.file_path = file_path
        self.dist = dist
        self.size = size
        self.file_name = os.path.basename(file_path).split('.')[0]
        self.dir_name = os.path.dirname(file_path)
        self.file_idx = self.file_name.split('_')[0]
        self.luma_par = os.path.join(self.dir_name,
                                     f'{self.file_idx}_luma_CTU.txt')
        self.chroma_par = os.path.join(self.dir_name,
                                      f'{self.file_idx}_chroma_CTU.txt')
        self.fwidth, self.fheight = self.size
        self.img = yuvio.imread(file, self.fwidth, self.fheight, 'yuv420p10le')
    
    def show_partition(self, comp:str):
        comp = comp.lower()
        if comp == 'y':
            img = self.img.y/1020
            fwidth, fheight = img.shape[1],img.shape[0]
            ctu_num = math.ceil(fwidth/128) * math.ceil(fheight/128)
            ctu_map = self.luma_par
            title = 'Luma CTU Partition'
        if comp == 'u':
            img = self.img.u/1020
            fwidth, fheight = img.shape[1],img.shape[0]
            ctu_num = math.ceil(fwidth/128) * math.ceil(fheight/128)
            ctu_map = self.chroma_par
            title = 'Chroma U CTU Partition'
        if comp == 'v':
            img = self.img.v/1020
            fwidth, fheight = img.shape[1],img.shape[0]
            ctu_num = math.ceil(fwidth/128) * math.ceil(fheight/128)
            ctu_map = self.chroma_par
            title = 'Chroma V CTU Partition'
        # for i in range(0, ctu_num):
        f = open(ctu_map, "r")
        for line in f:
            row = line.split(" ")
            startx , starty , height , width = int(row[0]) , int(row[1]), \
                                            int(row[2]) , int(row[3])
            if startx == 0 and starty==0 and height == 0 and width == 0:
                continue
            start_point = (startx, starty)
            end_point = (startx+width, starty+height)
            color =1
            thickness = 1
            frame_partition = cv2.rectangle(img, start_point, end_point,
                                            color, thickness)
        cv2.imshow(title, frame_partition)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_partition_map(self, comp:str)-> np.array:
        """Plot ctu partition
        file: image yuv file path
        size: image (width, height)
        ctu: ctu partition file (.txt) from vtm.
        component: Y, U, V
        """
        comp = comp.lower()
        if comp == 'y':
            img = self.img.y
            fwidth, fheight = img.shape[1],img.shape[0]
            frame_partition = np.ones((fheight,fwidth)) * 128
            ctu_num = math.ceil(fwidth/128) * math.ceil(fheight/128)
            ctu_map = self.luma_par
            title = os.path.join(self.dist,f'{self.file_idx}_luma_CTU.png')
        if comp in ['u','v']:
            img = self.img.u
            fwidth, fheight = img.shape[1],img.shape[0]
            frame_partition = np.ones((fheight,fwidth)) * 128
            ctu_num = math.ceil(fwidth/128) * math.ceil(fheight/128)
            ctu_map = self.luma_par
            title = os.path.join(self.dist,f'{self.file_idx}_chroma_CTU.png')
        f = open(ctu_map, "r")
        for line in f:
            row = line.split(" ")
            startx , starty , height , width = int(row[0]) , int(row[1]), \
                                            int(row[2]) , int(row[3])
            if startx == 0 and starty==0 and height == 0 and width == 0:
                continue
            start_point = (startx, starty)
            end_point = (startx+width, starty+height)
            color = 255
            thickness = 1
            frame_partition = cv2.rectangle(frame_partition, start_point, 
                                            end_point, color, thickness)
        # cv2.imshow(title, frame_partition)
        # cv2.waitKey()
        return title, frame_partition
    
    def save_img(self, title:str, img: np.array):
        cv2.imwrite(title, img)
        print(f'Save img {title} finished')
    
if __name__ == "__main__":
    file = '/home/woody/dataset/training_data/DIV2K_QP32_nDBF/0001_reco.yuv'
    partiton = CTU_Partition(file, size=(2040, 1404))
    title, map = partiton.get_partition_map('v')
    # print(map)
    img = Image.open('0001_chroma_CTU.png')
    print(np.array(img))
    # partiton.save_img(title, map)

    # partiton.save_partition_map('y')
