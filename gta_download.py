from PIL import Image
import requests
from io import BytesIO
import os
import shutil
import errno
import numpy as np
import matplotlib.pyplot as plt

M, N = 256, 256
val_size = 10

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gta")
original_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"gta_images")
if os.path.isdir(save_path):
    answer = input("Folder 'data/gta/' already exist. Are you sure you want to overrwrite it? [y/n]").lower()
    if answer == 'y' or answer == 'yes' or answer == '1':
        print("Removing old content...")
        shutil.rmtree(save_path)
    else:
        print("Cancelling...")
        exit(1)
print("Can't find gta dataset, making dataset")
os.makedirs(os.path.join(save_path, "input"))
os.makedirs(os.path.join(save_path, "input", "train"))
os.makedirs(os.path.join(save_path, "input", "val"))
os.makedirs(os.path.join(save_path, "input", "train","data"))
os.makedirs(os.path.join(save_path, "input", "val", "data"))
os.makedirs(os.path.join(save_path, "target1"))
os.makedirs(os.path.join(save_path, "target1", "train"))
os.makedirs(os.path.join(save_path, "target1", "train", "data"))
os.makedirs(os.path.join(save_path, "target1", "val"))
os.makedirs(os.path.join(save_path, "target1", "val","data"))
os.makedirs(os.path.join(save_path, "target2"))
os.makedirs(os.path.join(save_path, "target2", "val"))
os.makedirs(os.path.join(save_path, "target2", "val", "data"))
os.makedirs(os.path.join(save_path, "target2", "train"))
os.makedirs(os.path.join(save_path, "target2", "train", "data"))

images = []
print("Downloading images...")
IMAGE_NAMES = [
    "GTAV_ROADMAP_8192x8192.png",
    "GTAV_ATLUS_8192x8192.png",
    "GTAV_SATELLITE_8192x8192.png"
]
for image_name in IMAGE_NAMES:
    path = os.path.join(original_image_path, image_name)
    images.append(np.array(Image.open(path)))

print("Chopping them up into {}x{} images".format(M,N))
im0, im1, im2 = images
parted0 = [im0[x: x + M, y: y + N] for x in range(0, im0.shape[0], M) for y in range(0, im0.shape[1], N)]
parted1 = [im1[x: x + M, y: y + N] for x in range(0, im1.shape[0], M) for y in range(0, im1.shape[1], N)]
parted2 = [im2[x: x + M, y: y + N] for x in range(0, im2.shape[0], M) for y in range(0, im2.shape[1], N)]
idx = [i for i in range(len(parted0)) if not(parted0[i].mean(axis=(0,1))[2] > 160 and parted0[i].mean(axis=(0,1))[1] < 150 and parted0[i].mean(axis=(0,1))[0] < 100 or parted0[i].mean(axis=(0,1))[2] > 200)]
idx = [i for i in idx if not(parted0[i].var(axis=(0,1)).mean() < 100)]
val_idxs = np.random.choice(idx, val_size,replace=False)
print("Saving {}% of the complete image. Number of images: {}".format(int(100*(len(idx) / len(parted0))), len(idx)))
iters = 0
for i in idx:
    image_cat = "train"
    if i in val_idxs:
        image_cat = "val"
    im1 = Image.fromarray(parted0[i])
    im2 = Image.fromarray(parted1[i])
    im3 = Image.fromarray(parted2[i])

    im1.save(os.path.join(save_path, 'target1', image_cat, 'data', str(iters) + '_0.png'))
    im1.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_1.png"))  
    im1.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_2.png"))  
    im1.rotate(90).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_90_0.png"))
    im1.rotate(90).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_90_1.png"))
    im1.rotate(90).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_90_2.png"))
    im1.rotate(180).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_180.png"))
    im1.rotate(270).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_270_0.png"))  
    im1.rotate(270).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "target1",image_cat,"data", str(iters) + "_270_1.png"))


    im2.save(os.path.join(save_path, 'target2', image_cat, 'data', str(iters) + '_0.png'))
    im2.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_1.png"))  
    im2.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_2.png"))  
    im2.rotate(90).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_90_0.png"))
    im2.rotate(90).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_90_1.png"))
    im2.rotate(90).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_90_2.png"))
    im2.rotate(180).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_180.png"))
    im2.rotate(270).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_270_0.png"))  
    im2.rotate(270).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "target2",image_cat,"data", str(iters) + "_270_1.png"))

    im3.save(os.path.join(save_path, 'input', image_cat, 'data', str(iters) + '_0.png'))
    im3.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_1.png"))  
    im3.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_2.png"))  
    im3.rotate(90).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_90_0.png"))
    im3.rotate(90).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_90_1.png"))
    im3.rotate(90).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_90_2.png"))
    im3.rotate(180).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_180.png"))
    im3.rotate(270).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_270_0.png"))  
    im3.rotate(270).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_path, "input",image_cat,"data", str(iters) + "_270_1.png"))


    iters += 1


print("Generating chopped up image..")
im = np.zeros((8192,8192, 3))
j = 0
for row in range(8192//M):
    for col in range(8192//M):
        if j in idx:
            ims = parted1[j]
            im[row*M:M*(row+1), col*N:N*(col+1), :] = parted0[j]
        j+= 1

name = "GTA_choppedup.png"
path = os.path.join(original_image_path, name)

plt.imsave(path,im/255 )
print("Chopped up image saved in:", path)              


