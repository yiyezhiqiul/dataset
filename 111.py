import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io

img_path = '/media/totem_disk/totem/huang/SICAPv2/SICAPv2/masks'
img1_path = '/media/totem_disk/totem/huang/PANDA/Training_Images/0'

COLORS = np.array([
[100,100,100],
[0, 0, 0], #black
[0, 255, 255],#青色
[255, 0, 255], #purple
[128, 0, 0], #maroon
[255, 255, 0], #yellow
[255, 0, 0], #red
[0,0,255], #blue
[255, 255, 255],
], dtype=np.uint8)
for img in os.listdir(img_path):
    img1 = os.path.join(img_path,img)
    img2 = np.array(io.imread(img1))
    print(img2.shape)

    img3 = np.array(img2)[:,:,]

    img4 = COLORS[img3]
    plt.imshow(img4)
    plt.show()
    # img5 = io.imread(os.path.join(img1_path,img).replace('_mask.png','.jpg'))
    # plt.imshow(img5)
    # plt.show()
