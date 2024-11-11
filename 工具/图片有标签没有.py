import os

img_path ='/media/totem_disk/totem/huang/PANDA/Training_Images/0'
label_path ='/media/totem_disk/totem/huang/PANDA/Training_Labels/0'

z=0
for img in os.listdir(img_path):
    label_path1 = os.path.join(label_path,img).replace('.jpg','.png')
    if label_path1 not in os.listdir(label_path):
        # print(img)
        z=z+1
print(z)